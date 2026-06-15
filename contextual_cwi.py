import os
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, Set, Tuple, List
from wordfreq import zipf_frequency
from transformers import BertTokenizer, BertForMaskedLM

class ComplexWordIdentifier:
    """
    ComplexWordIdentifier assesses word complexity in context using a combination of
    BERT surprisal, word frequency, familiarity lists, and morphological features.
    """
    def __init__(self, config: Dict[str, Any], tokenizer: BertTokenizer, mlm_model: BertForMaskedLM, device: torch.device) -> None:
        """
        Initializes the CWI system with word lists, tokenizer, model, and device.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.mlm_model = mlm_model
        self.device = device
        
        # Load word lists for familiarity checks
        self.dale_chall_words = self._load_word_list(self.config['dale_chall_path'])
        self.oxford_words = self._load_word_list(self.config['oxford3000_path'])

    def _load_word_list(self, file_path: str) -> Set[str]:
        """
        Helper method to read word lists from file system.
        """
        words: Set[str] = set()
        if not os.path.exists(file_path):
            return words
        try:
            with open(file_path, 'r', encoding='utf-8') as file_obj:
                content = file_obj.read()
                # Handle UTF-8 BOM if present
                if content.startswith('\ufeff'):
                    content = content[1:]
                for line in content.splitlines():
                    word = line.strip().lower()
                    if word:
                        words.add(word)
        except Exception as exc:
            print(f"Error loading word list {file_path}: {exc}")
        return words

    @staticmethod
    def count_syllables(word: str) -> int:
        """
        Helper method to count syllables in a word using basic English heuristics.
        """
        word_lower = word.lower()
        if not word_lower:
            return 0
        vowels = "aeiouy"
        count = 0
        if word_lower[0] in vowels:
            count += 1
        for index in range(1, len(word_lower)):
            if word_lower[index] in vowels and word_lower[index - 1] not in vowels:
                count += 1
        if word_lower.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
        return count

    def get_surprisal(self, sentence: str, start_char: int, end_char: int, target_word: str) -> float:
        """
        Computes BERT surprisal: masks target word, computes MLM prob, and takes negative log prob.
        """
        masked_sentence = sentence[:start_char] + "[MASK]" + sentence[end_char:]
        
        try:
            inputs = self.tokenizer(
                masked_sentence, 
                max_length=self.config['max_bert_tokens'], 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            # Find [MASK] position
            mask_token_id = self.tokenizer.mask_token_id
            mask_indices = (inputs['input_ids'][0] == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_indices) == 0:
                return 0.5  # Neutral fallback
                
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
            
            # Target token mapping (handling wordpieces by taking the first one)
            target_tokens = self.tokenizer.tokenize(target_word)
            if not target_tokens:
                return 0.5
                
            target_token_id = self.tokenizer.convert_tokens_to_ids(target_tokens[0])
            prob = probs[target_token_id].item()
            
            # surprisal = -log10(prob)
            surprisal = -math.log10(max(1e-9, prob))
            # Normalize to 0-1 range based on standard maximum log surprisal (e.g., 9.0)
            normalized_surprisal = min(1.0, max(0.0, surprisal / 9.0))
            return normalized_surprisal
            
        except Exception as exc:
            print(f"Error calculating surprisal for '{target_word}': {exc}")
            return 0.5

    def get_complexity_score(self, sentence: str, start_char: int, end_char: int, target_word: str, lemma: str) -> float:
        """
        Computes context-aware complexity score using surprisal, frequency, familiarity, and morphology.
        """
        # 1. BERT Surprisal (weight 0.50)
        s_surprisal = self.get_surprisal(sentence, start_char, end_char, target_word)
        
        # 2. Frequency (weight 0.20)
        zipf = zipf_frequency(lemma, 'en')
        s_frequency = 1.0 - min(1.0, zipf / 8.0)
        
        # 3. Word Familiarity (weight 0.15)
        if lemma in self.dale_chall_words:
            s_familiarity = 0.0
        elif lemma in self.oxford_words:
            s_familiarity = 0.2
        else:
            s_familiarity = 1.0
            
        # 4. Morphological Features (weight 0.15)
        len_norm = min(1.0, len(target_word) / 15.0)
        syl = self.count_syllables(target_word)
        syl_norm = min(1.0, syl / 5.0)
        
        complex_suffixes = ('ification', 'ibility', 'ability', 'ness', 'ment', 'able', 'ious', 'ance', 'ence', 'tional', 'ative')
        suffix_boost = 0.2 if target_word.lower().endswith(complex_suffixes) else 0.0
        
        s_morphological = min(1.0, 0.4 * len_norm + 0.4 * syl_norm + 0.2 * suffix_boost)
        
        # Combined score calculation
        combined_score = (
            0.50 * s_surprisal +
            0.20 * s_frequency +
            0.15 * s_familiarity +
            0.15 * s_morphological
        )
        
        return combined_score

    def is_complex(self, sentence: str, start_char: int, end_char: int, target_word: str, lemma: str) -> Tuple[bool, float]:
        """
        Decides if a word is complex in the given context based on threshold.
        """
        score = self.get_complexity_score(sentence, start_char, end_char, target_word, lemma)
        is_comp = score >= self.config['simp_threshold']
        return is_comp, score
