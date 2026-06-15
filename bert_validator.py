# bert_validator.py

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_surprisal import BERTSurprisalCalculator

class BERTValidator:
    """
    BERTValidator performs 4 confidence checks to ensure candidate replacements
    preserve meaning, are preferred by BERT, are grammatically fluent,
    and offer a significant simplification gain.
    """
    def __init__(self, tokenizer=None, model=None, bert_model=None, device=None) -> None:
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = model if model is not None else BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.bert_model = bert_model if bert_model is not None else BertModel.from_pretrained('bert-base-uncased')
        
        self.surprisal_calc = BERTSurprisalCalculator(self.tokenizer, self.model, self.device)
        from bert_complexity import BERTComplexityScorer
        self.scorer = BERTComplexityScorer(self.tokenizer, self.model, self.bert_model, self.device)

    def get_sentence_embedding(self, sentence: str) -> torch.Tensor:
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            return outputs.last_hidden_state[0, 0]

    def get_word_embedding(self, sentence: str, word: str, start_char: int) -> torch.Tensor:
        inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
        with torch.no_grad():
            states = self.bert_model(**inputs).last_hidden_state[0]
        prefix_tokens = self.tokenizer.tokenize(sentence[:start_char])
        word_tokens = self.tokenizer.tokenize(word)
        start = min(len(prefix_tokens) + 1, states.size(0) - 1)
        end = min(start + len(word_tokens), states.size(0))
        return states[start:end].mean(dim=0)

    def compute_sentence_log_likelihood(self, sentence: str) -> float:
        """
        Measures grammatical fluency using BERT average cross entropy loss.
        """
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
        return -loss  # Higher (closer to 0) means more fluent

    def validate_replacement(
        self,
        sentence: str,
        original_word: str,
        candidate_word: str,
        start_char: int,
        end_char: int,
        pos_tag: str = None,
        debug: bool = False
    ) -> bool:
        """
        Runs the 4 confidence checks. Returns True if all pass.
        """
        # Prepare candidate sentence
        cand_sentence = sentence[:start_char] + candidate_word + sentence[end_char:]
        
        # 1. MLM Probability check (Check 1)
        masked_text = self.surprisal_calc.get_masked_sentence_and_idx(sentence, start_char, end_char)
        inputs = self.tokenizer(masked_text, return_tensors='pt').to(self.device)
        mask_idx = (inputs['input_ids'][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0][0].item()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        probs = F.softmax(logits[0, mask_idx], dim=-1)
        
        orig_tokens = self.tokenizer(original_word.lower(), add_special_tokens=False)['input_ids']
        cand_tokens = self.tokenizer(candidate_word.lower(), add_special_tokens=False)['input_ids']
        
        orig_prob = probs[orig_tokens[0]].item() if orig_tokens else 1e-9
        cand_prob = probs[cand_tokens[0]].item() if cand_tokens else 1e-9
        
        # Check 1: Candidate probability must be higher than original, OR sufficiently high on its own (> 0.001)
        if cand_prob <= orig_prob and cand_prob < 0.001:
            if debug:
                print(f"  [FAIL] Check 1 Fail: Candidate MLM prob ({cand_prob:.4f}) <= Original ({orig_prob:.4f}) and < 0.001")
            return False
            
        # 2. Meaning preservation check (Check 2)
        # Sentence-level CLS similarity
        orig_emb = self.get_sentence_embedding(sentence)
        cand_emb = self.get_sentence_embedding(cand_sentence)
        meaning_similarity = F.cosine_similarity(orig_emb.unsqueeze(0), cand_emb.unsqueeze(0)).item()
        
        # Word-level contextual embedding similarity
        orig_word_emb = self.get_word_embedding(sentence, original_word, start_char)
        cand_word_emb = self.get_word_embedding(cand_sentence, candidate_word, start_char)
        word_similarity = F.cosine_similarity(orig_word_emb.unsqueeze(0), cand_word_emb.unsqueeze(0)).item()
        
        if meaning_similarity <= 0.85 or word_similarity <= 0.65:
            if debug:
                print(f"  [FAIL] Check 2 Fail: Meaning similarity too low (Sent: {meaning_similarity:.4f} <= 0.85 or Word: {word_similarity:.4f} <= 0.65)")
            return False
            
        # 3. Fluency/Grammar check (Check 3)
        orig_fluency = self.compute_sentence_log_likelihood(sentence)
        cand_fluency = self.compute_sentence_log_likelihood(cand_sentence)
        
        # Must not decrease significantly (by more than 0.7 units of log loss)
        if cand_fluency - orig_fluency < -0.7:
            if debug:
                print(f"  [FAIL] Check 3 Fail: Grammatical degradation ({cand_fluency:.4f} -> {orig_fluency:.4f})")
            return False
            
        # 4. Improvement check (Check 4) - Contextual Complexity Reduction
        orig_complexity = self.scorer.compute_complexity_score(sentence, original_word, start_char, end_char, pos_tag)
        cand_complexity = self.scorer.compute_complexity_score(cand_sentence, candidate_word, start_char, start_char + len(candidate_word), pos_tag)
        complexity_reduction = orig_complexity - cand_complexity
        
        # Candidate must be simpler than original
        if complexity_reduction <= 0.0:
            if debug:
                print(f"  [FAIL] Check 4 Fail: Complexity reduction <= 0.0 ({complexity_reduction:.4f}) (Orig: {orig_complexity:.4f}, Cand: {cand_complexity:.4f})")
            return False
            
        if debug:
            print("  [PASS] All BERT Validator checks passed!")
            print(f"    MLM Pref: {cand_prob:.4f} > {orig_prob:.4f}")
            print(f"    Similarity: {meaning_similarity:.4f}")
            print(f"    Fluency Delta: {cand_fluency - orig_fluency:.4f}")
            print(f"    Complexity Red: {complexity_reduction:.4f}")
            
        return True
