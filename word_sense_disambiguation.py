import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel

class WordSenseDisambiguator:
    """
    WordSenseDisambiguator selects the correct WordNet synset/sense for a target word
    by comparing its contextual BERT embedding with embeddings from WordNet definitions/examples.
    """
    def __init__(self, config: Dict[str, Any], tokenizer: BertTokenizer, bert_model: BertModel, device: torch.device) -> None:
        """
        Initializes the WSD engine.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.device = device
        
        # Mapping spaCy POS to WordNet POS
        self.pos_map = {
            'NOUN': wn.NOUN,
            'VERB': wn.VERB,
            'ADJ': wn.ADJ,
            'ADV': wn.ADV,
            'PROPN': wn.NOUN
        }

    def get_word_embedding(self, sentence: str, start_char: int, end_char: int, target_word: str) -> Optional[torch.Tensor]:
        """
        Helper method to extract the contextual BERT embedding of a word in a sentence.
        """
        try:
            inputs = self.tokenizer(
                sentence, 
                max_length=self.config['max_bert_tokens'], 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                hidden_states = outputs.last_hidden_state[0] # Shape: [seq_len, 768]
                
            prefix_tokens = self.tokenizer.tokenize(sentence[:start_char])
            word_tokens = self.tokenizer.tokenize(target_word)
            
            start_token_idx = len(prefix_tokens) + 1  # Offset by 1 for [CLS]
            end_token_idx = start_token_idx + len(word_tokens)
            
            # Bound checking
            if end_token_idx > hidden_states.size(0):
                end_token_idx = hidden_states.size(0)
            if start_token_idx >= end_token_idx:
                return None
                
            # Average representations of wordpiece tokens
            word_embedding = hidden_states[start_token_idx:end_token_idx].mean(dim=0)
            return word_embedding
            
        except Exception as exc:
            print(f"Error extracting word embedding for '{target_word}': {exc}")
            return None

    def disambiguate(self, sentence: str, start_char: int, end_char: int, target_word: str, pos_tag: str) -> Optional[Any]:
        """
        Finds the WordNet synset that best matches the target word in context.
        """
        wn_pos = self.pos_map.get(pos_tag.upper())
        if not wn_pos:
            return None
            
        synsets = wn.synsets(target_word.lower(), pos=wn_pos)
        if not synsets:
            return None
            
        # 1. Get embedding of target word in current sentence context
        context_emb = self.get_word_embedding(sentence, start_char, end_char, target_word)
        if context_emb is None:
            return synsets[0]  # Fallback to first sense if contextual extraction fails
            
        best_sense = None
        highest_similarity = -1.0
        
        # 2. Iterate senses and compute cosine similarity
        for synset in synsets:
            examples = synset.examples()
            definition = synset.definition()
            
            contexts: List[str] = []
            # Gather all sentences: examples or mock definition context
            if examples:
                for example in examples:
                    # Only keep examples containing the target word (or variants) to extract context
                    if target_word.lower() in example.lower():
                        contexts.append(example)
                    else:
                        # Construct artificial context if target word is not explicitly in the example
                        contexts.append(f"{target_word} means {definition}.")
            else:
                contexts.append(f"{target_word} means {definition}.")
                
            sense_embeddings: List[torch.Tensor] = []
            for context in contexts[:3]: # Limit to first 3 contexts to ensure speed
                # Find start and end position of the target word inside the constructed/example context
                match = next(iter(self._find_word_positions(context, target_word)), None)
                if match:
                    st_ch, end_ch = match
                    emb = self.get_word_embedding(context, st_ch, end_ch, target_word)
                    if emb is not None:
                        sense_embeddings.append(emb)
                        
            if not sense_embeddings:
                # Direct average embedding fallback using definition text
                inputs = self.tokenizer(definition, max_length=self.config['max_bert_tokens'], padding=True, truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    emb = outputs.last_hidden_state[0, 0] # Use CLS embedding of definition
                sense_embeddings.append(emb)
                
            avg_sense_emb = torch.stack(sense_embeddings).mean(dim=0)
            similarity = F.cosine_similarity(context_emb, avg_sense_emb, dim=0).item()
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_sense = synset
                
        return best_sense if best_sense else synsets[0]

    def _find_word_positions(self, context: str, word: str) -> List[Tuple[int, int]]:
        """
        Helper method to locate the start and end indices of word in a context string.
        """
        positions: List[Tuple[int, int]] = []
        word_lower = word.lower()
        context_lower = context.lower()
        
        # Find exact matches
        for match in iter(self._regex_find_positions(context_lower, word_lower)):
            positions.append(match)
            
        # Try finding lemma form if exact match fails
        if not positions:
            for match in iter(self._regex_find_positions(context_lower, word_lower[:-1])):
                positions.append(match)
                
        return positions

    def _regex_find_positions(self, text: str, word: str) -> List[Tuple[int, int]]:
        import re
        matches: List[Tuple[int, int]] = []
        # Find matches bounded by word boundaries
        for match in re.finditer(r'\b' + re.escape(word) + r'\b', text):
            matches.append((match.start(), match.end()))
        return matches
