# mlm_scorer.py

import re
import torch
from transformers import BertTokenizer, BertForMaskedLM

class MLMScorer:
    """
    MLMScorer implements MLM probability calculations using BertForMaskedLM.
    """
    def __init__(self, tokenizer=None, model=None, device=None) -> None:
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained('bert-base-uncased')
        
        if model is not None:
            self.model = model
        else:
            self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            
        self.model.to(self.device)
        self.model.eval() # Step A - Verify model is in eval mode

    def get_candidate_probabilities(self, sentence: str, target_word: str, candidates: list) -> dict:
        """
        Step B, C, D: Correct masking and probability extraction.
        """
        mask_token = self.tokenizer.mask_token
        mask_id = self.tokenizer.mask_token_id
        
        # Step B - Correct masking using word boundaries
        masked_text = re.sub(
            r'\b' + re.escape(target_word) + r'\b',
            mask_token,
            sentence,
            count=1,
            flags=re.IGNORECASE
        )
        
        inputs = self.tokenizer(masked_text, return_tensors='pt').to(self.device)
        
        # Step C - Verify mask in tokenized input
        mask_positions = (inputs['input_ids'] == mask_id).nonzero()
        if len(mask_positions) == 0:
            return {c: 0.0001 for c in candidates}
            
        # Step D - Correct probability extraction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
        mask_pos = mask_positions[0][1].item()
        probs = torch.softmax(logits[0, mask_pos], dim=-1)
        
        results = {}
        for candidate in candidates:
            cand_tokens = self.tokenizer(candidate, add_special_tokens=False)['input_ids']
            if not cand_tokens:
                results[candidate] = 1e-9
                continue
            candidate_id = cand_tokens[0]
            results[candidate] = probs[candidate_id].item()
            
        return results

    def run_sanity_check(self, sentence: str, target_word: str) -> None:
        """
        Step E - Sanity check to print top 5 predictions for [MASK]
        """
        mask_token = self.tokenizer.mask_token
        mask_id = self.tokenizer.mask_token_id
        
        masked_text = re.sub(
            r'\b' + re.escape(target_word) + r'\b',
            mask_token,
            sentence,
            count=1,
            flags=re.IGNORECASE
        )
        
        inputs = self.tokenizer(masked_text, return_tensors='pt').to(self.device)
        mask_positions = (inputs['input_ids'] == mask_id).nonzero()
        if len(mask_positions) == 0:
            print(f"Sanity Check: [MASK] token not found for word '{target_word}' in '{sentence}'")
            return
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
        mask_pos = mask_positions[0][1].item()
        probs = torch.softmax(logits[0, mask_pos], dim=-1)
        
        top_5 = torch.topk(probs, 5)
        print(f"  Top 5 MLM predictions for [MASK] replacing '{target_word}':")
        for prob, idx in zip(top_5.values, top_5.indices):
            word = self.tokenizer.decode([idx])
            print(f"    {word}: {prob:.4f}")
