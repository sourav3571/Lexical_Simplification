import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Set, Optional
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertForMaskedLM
from wordfreq import zipf_frequency
import gensim.downloader as api
from gensim.models import KeyedVectors

# Global stop words list to prevent function words from becoming simplification candidates
STOP_WORDS: Set[str] = {
    'very', 'so', 'too', 'more', 'most', 'only', 'other', 'such', 'same', 'well',
    'just', 'not', 'no', 'this', 'that', 'these', 'those', 'who', 'whom', 'which',
    'what', 'how', 'why', 'where', 'when', 'then', 'there', 'here', 'all', 'any',
    'both', 'each', 'few', 'many', 'some', 'several', 'own', 'than', 'about',
    'above', 'after', 'again', 'against', 'along', 'among', 'around', 'at',
    'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
    'but', 'by', 'down', 'during', 'except', 'for', 'from', 'in', 'inside',
    'into', 'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past',
    'through', 'throughout', 'to', 'toward', 'under', 'underneath', 'until',
    'up', 'upon', 'with', 'within', 'without', 'and', 'or', 'nor', 'yet',
    'although', 'because', 'since', 'unless', 'while', 'whereas', 'the', 'a', 
    'an', 'is', 'was', 'were', 'be', 'been', 'being', 'am', 'are', 'it', 'its', 
    'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs'
}

def are_semantically_related(chosen_sense: Optional[Any], target_word: str, cand: str, pos_tag: str) -> bool:
    """
    Checks if a candidate is semantically and grammatically related to the target word.
    Ensures they share the same WordNet POS category and exist within the synset hierarchy.
    """
    pos_map = {
        'NOUN': wn.NOUN,
        'VERB': wn.VERB,
        'ADJ': wn.ADJ,
        'ADV': wn.ADV,
        'PROPN': wn.NOUN
    }
    wn_pos = pos_map.get(pos_tag.upper())
    if not wn_pos:
        return True
        
    c_s = wn.synsets(cand.lower(), pos=wn_pos)
    if not c_s:
        return False
        
    # Check relationship to the chosen_sense if WSD succeeded
    if chosen_sense:
        chosen_set = {chosen_sense}
        c_set = set(c_s)
        if chosen_set.intersection(c_set):
            return True
        
        # Check hypernyms/hyponyms/sisters of the chosen_sense
        if set(chosen_sense.hypernyms()).intersection(c_set):
            return True
        if set(chosen_sense.hyponyms()).intersection(c_set):
            return True
        for h in chosen_sense.hypernyms():
            if set(h.hyponyms()).intersection(c_set):
                return True
            if set(h.hypernyms()).intersection(c_set):
                return True
                
    # Fallback to general synset matching
    t_s = wn.synsets(target_word.lower(), pos=wn_pos)
    if not t_s:
        return False
        
    t_set = set(t_s)
    c_set = set(c_s)
    if t_set.intersection(c_set):
        return True
        
    for ts in t_s:
        if set(ts.hypernyms()).intersection(c_set):
            return True
        if set(ts.hyponyms()).intersection(c_set):
            return True
        for h in ts.hypernyms():
            if set(h.hyponyms()).intersection(c_set):
                return True
            if set(h.hypernyms()).intersection(c_set):
                return True
                
    return False


class CandidateGenerator:
    """
    CandidateGenerator generates lexical simplification candidates using
    WordNet, BERT MLM, and GloVe KNN, filtering for simplicity, POS, and semantic preservation.
    """
    def __init__(self, config: Dict[str, Any], tokenizer: BertTokenizer, mlm_model: BertForMaskedLM, device: torch.device) -> None:
        """
        Initializes sources, tokenizer, models, and loads GloVe embeddings.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.mlm_model = mlm_model
        self.device = device
        
        # Load GloVe Embeddings safely
        self.glove = None
        glove_name = self.config['glove_model']
        try:
            print(f"Loading GloVe Embeddings: {glove_name}...")
            self.glove = api.load(glove_name)
            print("GloVe model loaded successfully!")
        except Exception as exc:
            print(f"Warning: Failed to load GloVe model '{glove_name}'. Fallback to other sources: {exc}")

    def get_wordnet_candidates(self, chosen_sense: Optional[Any]) -> Set[str]:
        """
        Retrieves synonyms only from the active WordNet synset.
        """
        candidates: Set[str] = set()
        if not chosen_sense:
            return candidates
            
        for lemma in chosen_sense.lemmas():
            cand = lemma.name().replace('_', ' ').replace('-', ' ').strip().lower()
            # Ensure single words only
            if ' ' not in cand and '-' not in cand and cand.isalpha():
                candidates.add(cand)
        return candidates

    def get_mlm_candidates(self, sentence: str, start_char: int, end_char: int, top_n: int = 30) -> Set[str]:
        """
        Masks the word in the sentence and retrieves top context predictions from BERT.
        """
        candidates: Set[str] = set()
        masked_sentence = sentence[:start_char] + "[MASK]" + sentence[end_char:]
        
        try:
            inputs = self.tokenizer(
                masked_sentence, 
                max_length=self.config['max_bert_tokens'], 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            mask_token_id = self.tokenizer.mask_token_id
            mask_indices = (inputs['input_ids'][0] == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_indices) == 0:
                return candidates
                
            mask_idx = mask_indices[0].item()
            
            with torch.no_grad():
                if hasattr(self.mlm_model, 'mlm_head'):
                    bert_outputs = self.mlm_model.bert(**inputs)
                    hidden_states = bert_outputs.last_hidden_state[0, mask_idx]
                    logits = self.mlm_model.mlm_head(hidden_states)
                else:
                    outputs = self.mlm_model(**inputs)
                    logits = outputs.logits[0, mask_idx]
                probs = F.softmax(logits, dim=-1)
                
            top_indices = torch.topk(probs, k=top_n).indices.tolist()
            for idx in top_indices:
                tok = self.tokenizer.decode([idx]).strip().lower()
                if tok and not tok.startswith("##") and tok.isalpha():
                    candidates.add(tok)
                    
        except Exception as exc:
            print(f"Error generating MLM candidates: {exc}")
            
        return candidates

    def get_glove_candidates(self, word: str, top_n: int = 15) -> Set[str]:
        """
        Retrieves nearest neighbors of the target word from GloVe space.
        """
        candidates: Set[str] = set()
        if self.glove is None:
            return candidates
            
        word_lower = word.lower()
        if word_lower not in self.glove:
            return candidates
            
        try:
            similar_words = self.glove.most_similar(word_lower, topn=top_n)
            for sim_word, _ in similar_words:
                sim_word_lower = sim_word.strip().lower()
                if sim_word_lower.isalpha():
                    candidates.add(sim_word_lower)
        except Exception as exc:
            print(f"Error generating GloVe candidates: {exc}")
            
        return candidates

    def generate(self, sentence: str, start_char: int, end_char: int, target_word: str, pos_tag: str, chosen_sense: Optional[Any], cwi_engine: Any) -> List[Dict[str, Any]]:
        """
        Generates, merges, and filters candidates using all three sources.
        Implements progressive relaxation if no candidates pass filters.
        """
        word_lower = target_word.lower()
        
        # 1. Fetch raw candidates
        wn_cands = self.get_wordnet_candidates(chosen_sense)
        mlm_cands = self.get_mlm_candidates(sentence, start_char, end_char, top_n=30)
        glove_cands = self.get_glove_candidates(word_lower, top_n=15)
        
        all_raw_candidates = wn_cands.union(mlm_cands).union(glove_cands)
        
        # Calculate target word's complexity in context
        target_complexity = cwi_engine.get_complexity_score(sentence, start_char, end_char, target_word, word_lower)
        
        # Progressive threshold logic
        freq_thresholds = [self.config['freq_threshold'], 2.0, 0.0]
        simplicity_gating = [True, False]
        
        for sim_gate in simplicity_gating:
            for f_thresh in freq_thresholds:
                filtered: List[Dict[str, Any]] = []
                for cand in all_raw_candidates:
                    cand_lower = cand.lower()
                    
                    # Core criteria
                    if cand_lower == word_lower or cand_lower in STOP_WORDS:
                        continue
                    if not cand_lower.isalpha():
                        continue
                    if ' ' in cand_lower or '-' in cand_lower or '_' in cand_lower:
                        continue
                        
                    # Frequency check
                    zipf = zipf_frequency(cand_lower, 'en')
                    if zipf < f_thresh:
                        continue
                        
                    # POS & Semantic relation check
                    if not are_semantically_related(chosen_sense, target_word, cand_lower, pos_tag):
                        continue
                        
                    # Combined complexity check for candidate
                    cand_complexity = cwi_engine.get_complexity_score(sentence, start_char, end_char, cand_lower, cand_lower)
                    
                    if sim_gate and (cand_complexity >= target_complexity):
                        continue
                        
                    filtered.append({
                        'word': cand_lower,
                        'complexity': cand_complexity,
                        'simplicity_delta': target_complexity - cand_complexity
                    })
                    
                if filtered:
                    # Sort candidates by simplicity delta desc
                    filtered.sort(key=lambda x: x['simplicity_delta'], reverse=True)
                    return filtered
                    
        return []
