# bert_complexity.py

import math
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_surprisal import BERTSurprisalCalculator

class BERTComplexityScorer:
    """
    BERTComplexityScorer calculates a word's complexity using a weighted sum
    of surprisal (60%), top-10 prediction spread entropy (25%), and template embedding drift (15%).
    """
    def __init__(self, tokenizer=None, model=None, bert_model=None, device=None) -> None:
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = model if model is not None else BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.bert_model = bert_model if bert_model is not None else BertModel.from_pretrained('bert-base-uncased')
        
        self.model.to(self.device)
        self.bert_model.to(self.device)
        self.model.eval()
        self.bert_model.eval()
        
        self.surprisal_calc = BERTSurprisalCalculator(self.tokenizer, self.model, self.device)

    def get_word_embedding(self, sentence: str, start_char: int, end_char: int, target_word: str) -> torch.Tensor:
        """
        Extracts contextual BERT embedding for a target word.
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

    def get_prediction_spread(self, sentence: str, start_char: int, end_char: int) -> float:
        """
        Score 2: prediction spread entropy.
        """
        masked_text = self.surprisal_calc.get_masked_sentence_and_idx(sentence, start_char, end_char)
        inputs = self.tokenizer(masked_text, return_tensors='pt').to(self.device)
        mask_indices = (inputs['input_ids'][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_indices) == 0:
            return 0.5
        mask_idx = mask_indices[0].item()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        probs = F.softmax(logits[0, mask_idx], dim=-1)
        top_10 = torch.topk(probs, 10)
        top_probs = top_10.values
        
        sum_probs = top_probs.sum()
        if sum_probs > 0:
            norm_probs = top_probs / sum_probs
        else:
            norm_probs = torch.ones(10) / 10.0
            
        entropy = -torch.sum(norm_probs * torch.log(norm_probs + 1e-9)).item()
        normalized_entropy = entropy / 2.3026  # norm by ln(10)
        return normalized_entropy

    def get_embedding_drift(self, sentence: str, start_char: int, end_char: int, target_word: str) -> float:
        """
        Score 3: Embedding drift compared to template sentences.
        """
        # Current contextual embedding
        curr_emb = self.get_word_embedding(sentence, start_char, end_char, target_word)
        
        # Template sentence embedding
        template = f"I see the {target_word}."
        t_start = template.find(target_word)
        t_end = t_start + len(target_word)
        temp_emb = self.get_word_embedding(template, t_start, t_end, target_word)
        
        if curr_emb.sum() == 0 or temp_emb.sum() == 0:
            return 0.5
            
        cosine_sim = F.cosine_similarity(curr_emb.unsqueeze(0), temp_emb.unsqueeze(0)).item()
        drift = 1.0 - max(-1.0, min(1.0, cosine_sim))
        # Normalize drift to [0.0, 1.0] (since drift is in [0, 2], map to [0, 1])
        return drift / 2.0

    def get_inherent_complexity(self, target_word: str) -> float:
        """
        Computes the inherent complexity of a word based on its BERT MLM prediction bias
        and a subword token length penalty.
        """
        bias = self.model.cls.predictions.bias
        toks = self.tokenizer.encode(target_word, add_special_tokens=False)
        if not toks:
            return 1.0
            
        # Average bias across subword tokens
        biases = [bias[t].item() for t in toks]
        avg_bias = sum(biases) / len(biases)
        
        # Map bias to a complexity score between 0.0 and 1.0
        # Biases typically range from -1.5 to 0.5. Map -1.0 to 0.5 -> 1.0 to 0.0
        mapped_complexity = 1.0 - (avg_bias + 1.0) / 1.5
        mapped_complexity = max(0.0, min(1.0, mapped_complexity))
        
        # Add subword length penalty (representing rare/complex morphological structure)
        token_penalty = 0.15 * (len(toks) - 1)
        
        return min(1.0, mapped_complexity + token_penalty)

    def get_neutral_template_surprisal(self, target_word: str, pos: str = None) -> float:
        """
        Legacy method kept for backward compatibility; routes to inherent complexity.
        """
        return self.get_inherent_complexity(target_word)

    def compute_complexity_score(self, sentence: str, target_word: str, start_char: int, end_char: int, pos: str = None) -> float:
        """
        Returns the combined complexity score using:
          - 45% inherent complexity (BERT MLM bias + subword length, main rarity signal)
          - 30% embedding drift (context vs. template, captures figurative/abstract use)
          - 25% prediction spread entropy (lexical uncertainty at mask position)

        Rebalanced so vocabulary rarity (inherent) is the dominant term,
        preventing words like 'physician'/'comprehended' from being under-scored.
        """
        inherent = self.get_inherent_complexity(target_word)
        drift    = self.get_embedding_drift(sentence, start_char, end_char, target_word)
        spread   = self.get_prediction_spread(sentence, start_char, end_char)

        score = 0.45 * inherent + 0.30 * drift + 0.25 * spread
        return min(1.0, score)
