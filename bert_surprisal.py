# bert_surprisal.py

import math
import re
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM

class BERTSurprisalCalculator:
    """
    BERTSurprisalCalculator uses BERT Masked Language Modeling to compute
    the contextual surprisal of a target word.
    """
    def __init__(self, tokenizer=None, model=None, device=None) -> None:
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = model if model is not None else BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model.to(self.device)
        self.model.eval()

    def get_masked_sentence_and_idx(self, sentence: str, start_char: int, end_char: int) -> tuple:
        """
        Masks the target word in the sentence.

        If the target word is at the end of the sentence and there is no
        terminal punctuation, append a period so BERT sees a well-formed
        sentence and does not over-predict punctuation tokens.
        """
        sentence = sentence.rstrip()
        prefix = sentence[:start_char]
        suffix = sentence[end_char:]
        if end_char == len(sentence) and sentence and sentence[-1] not in {'.', '!', '?'}:
            suffix = suffix + '.'
        masked_text = prefix + self.tokenizer.mask_token + suffix
        return masked_text

    def compute_surprisal(self, sentence: str, target_word: str, start_char: int, end_char: int) -> float:
        """
        Computes surprisal = -log(probability) of the target word in context.
        Returns normalized surprisal between 0.0 and 1.0.
        """
        # Append period if sentence doesn't end with punctuation
        sentence_clean = sentence.strip()
        has_punctuation = len(sentence_clean) > 0 and sentence_clean[-1] in {'.', '!', '?'}
        if not has_punctuation:
            sentence = sentence_clean + "."
            
        masked_text = self.get_masked_sentence_and_idx(sentence, start_char, end_char)
        inputs = self.tokenizer(masked_text, return_tensors='pt').to(self.device)
        
        mask_token_id = self.tokenizer.mask_token_id
        mask_indices = (inputs['input_ids'][0] == mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_indices) == 0:
            return 0.5
            
        mask_idx = mask_indices[0].item()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        probs = F.softmax(logits[0, mask_idx], dim=-1)
        
        # Tokenize target word
        target_tokens = self.tokenizer(target_word, add_special_tokens=False)['input_ids']
        if not target_tokens:
            return 1.0
            
        target_id = target_tokens[0]
        prob = probs[target_id].item()
        
        # Surprisal calculation
        surprisal = -math.log(max(1e-9, prob))
        # Normalize surprisal: log(1e-9) is ~20.7. Let's map [0, 15] to [0.0, 1.0]
        normalized_surprisal = min(1.0, max(0.0, surprisal / 15.0))
        return normalized_surprisal

    def run_sanity_check(self, sentence: str, target_word: str, start_char: int, end_char: int) -> None:
        """
        Prints the top 5 predictions for [MASK].
        """
        masked_text = self.get_masked_sentence_and_idx(sentence, start_char, end_char)
        inputs = self.tokenizer(masked_text, return_tensors='pt').to(self.device)
        mask_indices = (inputs['input_ids'][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        
        if len(mask_indices) == 0:
            print(f"Sanity Check: [MASK] not found for '{target_word}' in '{sentence}'")
            return
            
        mask_idx = mask_indices[0].item()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        probs = F.softmax(logits[0, mask_idx], dim=-1)
        top_5 = torch.topk(probs, 5)
        
        print(f"\nTop 5 MLM predictions for mask in sentence: '{sentence}'")
        print(f"Replacing target word: '{target_word}'")
        for prob, idx in zip(top_5.values, top_5.indices):
            decoded_word = self.tokenizer.decode([idx])
            print(f"  {decoded_word}: {prob:.4f}")
