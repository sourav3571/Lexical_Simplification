# bert_sense_disambiguator.py

import torch
import torch.nn.functional as F
from typing import Any
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel

class BERTSenseDisambiguator:
    """
    BERTSenseDisambiguator selects the most appropriate WordNet synset
    by measuring cosine similarity between the target word's contextual embedding
    and the embeddings of WordNet definitions/examples.
    """
    def __init__(self, tokenizer=None, bert_model=None, device=None) -> None:
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = bert_model if bert_model is not None else BertModel.from_pretrained('bert-base-uncased')
        
        self.bert_model.to(self.device)
        self.bert_model.eval()

        self.pos_map = {
            'NOUN': wn.NOUN,
            'VERB': wn.VERB,
            'ADJ': wn.ADJ,
            'ADV': wn.ADV,
            'PROPN': wn.NOUN
        }

    def get_word_embedding(self, sentence: str, start_char: int, end_char: int, target_word: str) -> torch.Tensor:
        """
        Retrieves the average wordpiece token embedding for the target word in context.
        """
        inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            hidden_states = outputs.last_hidden_state[0]
            
        prefix_tokens = self.tokenizer.tokenize(sentence[:start_char])
        word_tokens = self.tokenizer.tokenize(target_word)
        
        start_idx = min(len(prefix_tokens) + 1, hidden_states.size(0) - 1)
        end_idx = min(start_idx + len(word_tokens), hidden_states.size(0))
        if start_idx >= end_idx:
            return torch.zeros(768).to(self.device)
            
        return hidden_states[start_idx:end_idx].mean(dim=0)

    def disambiguate(self, sentence: str, start_char: int, end_char: int, target_word: str, pos_tag: str) -> tuple:
        """
        Finds the WordNet synset with the highest contextual embedding similarity.
        Returns a tuple: (best_sense, highest_sim)
        """
        wn_pos = self.pos_map.get(pos_tag.upper())
        if not wn_pos:
            synsets = wn.synsets(target_word.lower())
        else:
            synsets = wn.synsets(target_word.lower(), pos=wn_pos)
            
        if not synsets:
            synsets = wn.synsets(target_word.lower())
            if not synsets:
                return None, 0.0
                
        context_emb = self.get_word_embedding(sentence, start_char, end_char, target_word)
        
        best_sense = synsets[0]
        highest_sim = -1.0
        
        for synset in synsets:
            definition = synset.definition()
            temp_sentence = f"{target_word} means {definition}."
            t_start = temp_sentence.find(target_word)
            t_end = t_start + len(target_word)
            
            def_emb = self.get_word_embedding(temp_sentence, t_start, t_end, target_word)
            
            sim = F.cosine_similarity(context_emb.unsqueeze(0), def_emb.unsqueeze(0)).item()
            if sim > highest_sim:
                highest_sim = sim
                best_sense = synset
                
        return best_sense, highest_sim
