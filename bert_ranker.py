# bert_ranker.py

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_surprisal import BERTSurprisalCalculator

class GatedFusionRanker(nn.Module):
    """
    GatedFusionRanker is a neural ranker that maps 4 BERT-based contextual features:
    1. MLM fit probability
    2. Contextual embedding cosine similarity
    3. Surprisal reduction
    4. Sentence fluency change
    to a single score in [0.0, 1.0].
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights to act as a sensible default linear combination
        # features: [mlm, cosine, surprisal_red, fluency_change]
        with torch.no_grad():
            self.net[0].weight.copy_(torch.tensor([[0.2, 4.0, 0.3, 0.3]], dtype=torch.float32))
            self.net[0].bias.fill_(-3.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def predict(self, mlm_prob: float, cosine_sim: float, surp_red: float, fluency_change: float) -> float:
        """
        Predicts ranking score for a single candidate.
        """
        feats = torch.tensor([[mlm_prob, cosine_sim, surp_red, fluency_change]], dtype=torch.float32)
        with torch.no_grad():
            score = self.forward(feats).item()
        return score

    def train_on_lex_mturk(
        self,
        file_path: str,
        tokenizer: BertTokenizer,
        model: BertForMaskedLM,
        bert_model: BertModel,
        device: torch.device,
        limit: int = 5
    ) -> None:
        """
        Trains the ranker network on the LexMTurk dataset using BERT feature extraction.
        """
        if not os.path.exists(file_path):
            print(f"Skipping training: {file_path} not found.")
            return

        print(f"Starting LexMTurk ranker training on {limit} sentences...")
        self.to(device)
        self.train()
        
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        surp_calc = BERTSurprisalCalculator(tokenizer, model, device)
        
        # Parse LexMTurk / BenchLS
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading dataset file: {e}")
            return
            
        trained_samples = 0
        for line in lines:
            if trained_samples >= limit:
                break
                
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
                
            sentence = parts[0]
            target = parts[1]
            
            # Clean quote markers if present
            if sentence.startswith('"') and sentence.endswith('"'):
                sentence = sentence[1:-1].strip()
                
            match = re.search(r'\b' + re.escape(target) + r'\b', sentence, re.IGNORECASE)
            if not match:
                continue
                
            start_char = match.start()
            end_char = match.end()
            
            # Parse candidates and their frequencies
            candidates = []
            start_idx = 3
            if len(parts) > 2:
                if not parts[2].isdigit():
                    start_idx = 2
                    
            cand_items = parts[start_idx:]
            has_colons = any(':' in item for item in cand_items)
            if has_colons:
                for item in cand_items:
                    if ':' in item:
                        item_parts = item.split(':')
                        if len(item_parts) >= 2:
                            try:
                                cand = item_parts[0].strip().lower()
                                votes = int(item_parts[1])
                                candidates.append((cand, votes))
                            except ValueError:
                                pass
            else:
                from collections import Counter
                votes_counter = Counter([item.strip().lower() for item in cand_items if item.strip()])
                candidates = [(cand, votes) for cand, votes in votes_counter.items()]
                        
            if not candidates:
                continue
                
            # Get max votes to normalize target labels
            max_votes = max(c[1] for c in candidates)
            if max_votes == 0:
                max_votes = 1
                
            # Calculate BERT features for candidates
            # We will use this to generate training pairs
            orig_surp = surp_calc.compute_surprisal(sentence, target, start_char, end_char)
            
            # Context embedding of original sentence
            orig_inputs = tokenizer(sentence, return_tensors='pt').to(device)
            with torch.no_grad():
                orig_outputs = bert_model(**orig_inputs)
                orig_states = orig_outputs.last_hidden_state[0]
            prefix_tokens = tokenizer.tokenize(sentence[:start_char])
            word_tokens = tokenizer.tokenize(target)
            orig_start = min(len(prefix_tokens) + 1, orig_states.size(0) - 1)
            orig_end = min(orig_start + len(word_tokens), orig_states.size(0))
            orig_word_emb = orig_states[orig_start:orig_end].mean(dim=0)
            
            # Pre-calculate fluency of original
            orig_inputs_full = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(device)
            with torch.no_grad():
                orig_loss = model(**orig_inputs_full, labels=orig_inputs_full['input_ids']).loss.item()
            orig_fluency = -orig_loss
            
            # MLM probabilities for candidates in batch
            masked_text = surp_calc.get_masked_sentence_and_idx(sentence, start_char, end_char)
            mask_inputs = tokenizer(masked_text, return_tensors='pt').to(device)
            mask_idx = (mask_inputs['input_ids'][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0][0].item()
            with torch.no_grad():
                mask_outputs = model(**mask_inputs)
                mask_logits = mask_outputs.logits
            probs = F.softmax(mask_logits[0, mask_idx], dim=-1)
            
            for cand, votes in candidates[:3]:  # Top 3 candidates per sentence for speed
                # 1. MLM
                cand_toks = tokenizer(cand, add_special_tokens=False)['input_ids']
                if not cand_toks:
                    continue
                mlm_prob = probs[cand_toks[0]].item()
                
                # 2. Embedding Cosine
                cand_sentence = sentence[:start_char] + cand + sentence[end_char:]
                cand_inputs = tokenizer(cand_sentence, return_tensors='pt').to(device)
                with torch.no_grad():
                    cand_outputs = bert_model(**cand_inputs)
                    cand_states = cand_outputs.last_hidden_state[0]
                cand_prefix = tokenizer.tokenize(sentence[:start_char])
                cand_toks_list = tokenizer.tokenize(cand)
                cand_start = min(len(cand_prefix) + 1, cand_states.size(0) - 1)
                cand_end = min(cand_start + len(cand_toks_list), cand_states.size(0))
                cand_word_emb = cand_states[cand_start:cand_end].mean(dim=0)
                
                cosine_sim = F.cosine_similarity(orig_word_emb.unsqueeze(0), cand_word_emb.unsqueeze(0)).item()
                
                # 3. Surprisal reduction
                cand_surp = surp_calc.compute_surprisal(sentence, cand, start_char, end_char)
                surp_red = orig_surp - cand_surp
                
                # 4. Fluency change
                cand_inputs_full = tokenizer(cand_sentence, return_tensors='pt', padding=True, truncation=True).to(device)
                with torch.no_grad():
                    cand_loss = model(**cand_inputs_full, labels=cand_inputs_full['input_ids']).loss.item()
                cand_fluency = -cand_loss
                fluency_change = cand_fluency - orig_fluency
                
                # Forward, loss, backward
                features_tensor = torch.tensor([[mlm_prob, cosine_sim, surp_red, fluency_change]], dtype=torch.float32).to(device)
                target_score = torch.tensor([[votes / max_votes]], dtype=torch.float32).to(device)
                
                optimizer.zero_grad()
                pred_score = self.forward(features_tensor)
                loss = criterion(pred_score, target_score)
                loss.backward()
                optimizer.step()
                
            trained_samples += 1
            print(f"  Sentence {trained_samples}/{limit} trained successfully.")
            
        self.eval()
        print("Training complete! Model is now optimized.")
