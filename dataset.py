import os
import re
import math
import torch
import spacy
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple, Optional
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F
from wordfreq import zipf_frequency

class LexicalSimplificationDataset(Dataset):
    """
    LexicalSimplificationDataset parses target simplification datasets (LexMTurk/BenchLS)
    and represents them as PyTorch-compatible raw samples.
    """
    def __init__(self, config: Dict[str, Any], tokenizer: BertTokenizer, data_path: str) -> None:
        """
        Initializes dataset paths and loads parsing models.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.data_path = data_path
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        self.samples = self._load_or_generate_dataset()

    def _load_or_generate_dataset(self) -> List[Dict[str, Any]]:
        """
        Checks for BenchLS or LexMTurk file, parses it, or raises FileNotFoundError.
        """
        target_path = self.data_path
        if os.path.isdir(target_path):
            benchls_path = os.path.join(target_path, "BenchLS.txt")
            lex_mturk_path = os.path.join(target_path, "lex_mturk.txt")
            if os.path.exists(benchls_path):
                file_to_parse = benchls_path
            elif os.path.exists(lex_mturk_path):
                file_to_parse = lex_mturk_path
            else:
                raise FileNotFoundError(f"Neither BenchLS.txt nor lex_mturk.txt found in {target_path}")
        else:
            if os.path.exists(target_path):
                file_to_parse = target_path
            else:
                raise FileNotFoundError(f"Dataset path {target_path} does not exist.")

        print(f"Parsing dataset from {file_to_parse}...")
        return self._parse_file(file_to_parse)

    def _parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses LexMTurk / BenchLS formats into structured candidate/label training samples.
        """
        samples: List[Dict[str, Any]] = []
        with open(file_path, 'r', encoding='utf-8') as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 4:
                    continue
                    
                sentence = parts[0]
                target = parts[1]
                
                try:
                    position = int(parts[2])
                except ValueError:
                    continue
                    
                words = sentence.split(' ')
                if position >= len(words):
                    match = re.search(r'\b' + re.escape(target) + r'\b', sentence, re.IGNORECASE)
                    if not match:
                        continue
                    start_char = match.start()
                    end_char = match.end()
                else:
                    prefix = " ".join(words[:position])
                    start_char = len(prefix) + 1 if position > 0 else 0
                    end_char = start_char + len(target)
                
                # Determine POS of target in sentence context
                target_pos = 'NOUN'
                try:
                    doc = self.nlp(sentence)
                    for token in doc:
                        if token.idx == start_char or (token.idx <= start_char and token.idx + len(token.text) >= end_char):
                            target_pos = token.pos_
                            break
                except Exception:
                    pass
                
                # Extract candidates and votes
                candidate_parts = parts[3:]
                parsed_candidates: List[Tuple[str, int]] = []
                max_votes = 1
                
                for item in candidate_parts:
                    if ':' in item:
                        parts_item = item.split(':')
                        if len(parts_item) == 2:
                            left, right = parts_item
                            try:
                                votes = int(right)
                                candidate = left
                                parsed_candidates.append((candidate, votes))
                                max_votes = max(max_votes, votes)
                            except ValueError:
                                try:
                                    votes = int(left)
                                    candidate = right
                                    parsed_candidates.append((candidate, votes))
                                    max_votes = max(max_votes, votes)
                                except ValueError:
                                    continue
                                    
                if not parsed_candidates:
                    continue
                    
                for cand_word, votes in parsed_candidates:
                    label = float(votes) / float(max_votes)
                    samples.append({
                        'sentence': sentence,
                        'target_word': target,
                        'target_pos': target_pos,
                        'start_char': start_char,
                        'end_char': end_char,
                        'candidate': cand_word,
                        'label': label
                    })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class PrecomputedDataset(Dataset):
    """
    Simple wrapper dataset containing precomputed tensors to optimize PyTorch training loops.
    """
    def __init__(self, cached_features: List[Dict[str, Any]]) -> None:
        self.features = cached_features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.features[idx]


def precompute_features(
    samples: List[Dict[str, Any]], 
    config: Dict[str, Any], 
    tokenizer: BertTokenizer, 
    model: nn.Module, 
    cwi_engine: Any, 
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Extracts BERT and morphological complexity features for training rankers.
    Precomputation avoids slow model forward passes inside training loops.
    """
    precomputed: List[Dict[str, Any]] = []
    
    # Pre-load lists for fast lookup
    dale_chall = cwi_engine.dale_chall_words
    oxford = cwi_engine.oxford_words
    
    # Group samples by sentence to minimize BERT passes on the same sentence
    grouped_samples: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for index, sample in enumerate(samples):
        sent = sample['sentence']
        if sent not in grouped_samples:
            grouped_samples[sent] = []
        grouped_samples[sent].append((index, sample))
        
    cache: List[Dict[str, Any]] = []
    
    for sentence, group in grouped_samples.items():
        # 1. Tokenize sentence and get BERT contextual embeddings
        encoded_sent = tokenizer(
            sentence, 
            max_length=config['max_length'], 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model.bert(input_ids=encoded_sent['input_ids'], attention_mask=encoded_sent['attention_mask'])
            context_embed = outputs.last_hidden_state[0, 0].cpu() # CLS token (768-dim)
            
        # Group by target word within the same sentence
        word_groups: Dict[Tuple[str, int, int], List[Tuple[int, Dict[str, Any]]]] = {}
        for index, sample in group:
            key = (sample['target_word'], sample['start_char'], sample['end_char'])
            if key not in word_groups:
                word_groups[key] = []
            word_groups[key].append((index, sample))
            
        for (target_word, start_char, end_char), group_items in word_groups.items():
            # Get original word contextual embedding
            prefix_tokens = tokenizer.tokenize(sentence[:start_char])
            word_tokens = tokenizer.tokenize(target_word)
            
            orig_start_idx = len(prefix_tokens) + 1
            orig_end_idx = orig_start_idx + len(word_tokens)
            
            orig_seq_len = outputs.last_hidden_state.size(1)
            orig_start_idx_c = min(orig_start_idx, orig_seq_len - 1)
            orig_end_idx_c = min(max(orig_end_idx, orig_start_idx_c + 1), orig_seq_len)
            
            orig_contextual_embed = outputs.last_hidden_state[0, orig_start_idx_c:orig_end_idx_c].mean(dim=0)
            
            # Mask the target word position to get MLM probabilities
            masked_sentence = sentence[:start_char] + "[MASK]" + sentence[end_char:]
            masked_inputs = tokenizer(
                masked_sentence, 
                max_length=config['max_length'], 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            ).to(device)
            
            mask_token_id = tokenizer.mask_token_id
            mask_indices = (masked_inputs['input_ids'][0] == mask_token_id).nonzero(as_tuple=True)[0]
            
            if len(mask_indices) > 0:
                mask_idx = mask_indices[0].item()
                with torch.no_grad():
                    masked_outputs = model.bert(input_ids=masked_inputs['input_ids'], attention_mask=masked_inputs['attention_mask'])
                    masked_features = masked_outputs.last_hidden_state[0, mask_idx]
                    mlm_logits = model.mlm_head(masked_features)
                    mlm_probs = F.softmax(mlm_logits, dim=-1)
            else:
                mlm_probs = torch.zeros(tokenizer.vocab_size).to(device)
                
            # Compute complexity of original word
            target_complexity = cwi_engine.get_complexity_score(sentence, start_char, end_char, target_word, target_word.lower())
            
            # Process candidate specific features
            for idx, sample in group_items:
                candidate_word = sample['candidate']
                cand_lower = candidate_word.lower()
                
                cand_tokens = tokenizer.tokenize(candidate_word)
                cand_token_id = tokenizer.convert_tokens_to_ids(cand_tokens[0]) if cand_tokens else tokenizer.unk_token_id
                
                mlm_prob_val = mlm_probs[cand_token_id].item()
                
                # Get contextual embedding for candidate in substituted sentence
                cand_sentence = sentence[:start_char] + candidate_word + sentence[end_char:]
                cand_encoded = tokenizer(
                    cand_sentence, 
                    max_length=config['max_length'], 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt'
                ).to(device)
                
                with torch.no_grad():
                    cand_outputs = model.bert(input_ids=cand_encoded['input_ids'], attention_mask=cand_encoded['attention_mask'])
                    
                cand_prefix_tokens = tokenizer.tokenize(sentence[:start_char])
                cand_word_tokens = tokenizer.tokenize(candidate_word)
                cand_start_idx = len(cand_prefix_tokens) + 1
                cand_end_idx = cand_start_idx + len(cand_word_tokens)
                
                cand_seq_len = cand_outputs.last_hidden_state.size(1)
                cand_start_idx_c = min(cand_start_idx, cand_seq_len - 1)
                cand_end_idx_c = min(max(cand_end_idx, cand_start_idx_c + 1), cand_seq_len)
                cand_contextual_embed = cand_outputs.last_hidden_state[0, cand_start_idx_c:cand_end_idx_c].mean(dim=0)
                
                cosine_sim = F.cosine_similarity(orig_contextual_embed.unsqueeze(0), cand_contextual_embed.unsqueeze(0)).item()
                
                # Compute complexity of candidate
                cand_complexity = cwi_engine.get_complexity_score(sentence, start_char, end_char, cand_lower, cand_lower)
                simplicity_delta = target_complexity - cand_complexity
                
                cache.append({
                    'index': idx,
                    'context_embed': context_embed,
                    'mlm_prob': torch.tensor([mlm_prob_val], dtype=torch.float32),
                    'semantic_similarity': torch.tensor([cosine_sim], dtype=torch.float32),
                    'simplicity_delta': torch.tensor([simplicity_delta], dtype=torch.float32),
                    'label': torch.tensor(sample['label'], dtype=torch.float32)
                })
                
        # Update progress prints for large files
        if len(cache) % 1000 == 0 and len(cache) > 0:
            print(f"Precomputed {len(cache)} samples...")
            
    cache.sort(key=lambda x: x['index'])
    return cache
