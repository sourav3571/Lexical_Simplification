import os
import sys
import random
import re
import math
import logging
import json
import gc
from typing import List, Dict, Any, Tuple, Set, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import spacy
import nltk
from nltk.corpus import wordnet as wn
# nltk data should be cached locally
import gensim.downloader as api
from gensim.models import KeyedVectors
import wordfreq
from wordfreq import zipf_frequency

from transformers import BertTokenizer, BertModel, BertForMaskedLM, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Global config
CONFIG = {
    'bert_model'    : 'bert-base-uncased',
    'glove_model'   : 'glove-wiki-gigaword-100',
    'batch_size'    : 16,
    'epochs'        : 10,  # 10 epochs is sufficient and very fast on CPU with caching
    'lr_bert'       : 2e-5,
    'lr_ranker'     : 1e-3,
    'max_length'    : 32,
    'dropout'       : 0.3,
    'freq_threshold': 4.0,
    'simp_threshold': 4.5,
    'glove_dim'     : 100,
    'seed'          : 42,
    'val_split'     : 0.1,
    'grad_clip'     : 1.0,
    'weight_decay'  : 0.01,
    'fine_tune_bert': False,
}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG['seed'])

# ---------------------------------------------------------
# Load Word Lists (Dale-Chall & Oxford 3000)
# ---------------------------------------------------------
def load_word_lists() -> Tuple[Set[str], Set[str]]:
    dale_chall_words = set()
    oxford_words = set()
    
    # Check parent dirs or current workspace path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    dc_path = os.path.join(base_dir, "dale_chall.txt")
    if not os.path.exists(dc_path):
        dc_path = "dale_chall.txt"
        
    if os.path.exists(dc_path):
        with open(dc_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    dale_chall_words.add(word)
        print(f"Loaded {len(dale_chall_words)} Dale-Chall easy words.")
    else:
        print("Warning: dale_chall.txt not found. Familiarity matching will have fallbacks.")
        
    ox_path = os.path.join(base_dir, "oxford3000.txt")
    if not os.path.exists(ox_path):
        ox_path = "oxford3000.txt"
        
    if os.path.exists(ox_path):
        with open(ox_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if content.startswith('\ufeff'):
                content = content[1:]
            for line in content.splitlines():
                word = line.strip().lower()
                if word:
                    oxford_words.add(word)
        print(f"Loaded {len(oxford_words)} Oxford 3000 words.")
    else:
        print("Warning: oxford3000.txt not found. Familiarity matching will have fallbacks.")
        
    return dale_chall_words, oxford_words

# Load global lists
DALE_CHALL_WORDS, OXFORD_WORDS = load_word_lists()

# ---------------------------------------------------------
# Helper: BERT surprisal computation
# ---------------------------------------------------------
def compute_bert_surprisal(sentence: str, start_char: int, end_char: int, target_word: str, bert_model: nn.Module, tokenizer: BertTokenizer, device: torch.device) -> float:
    masked_sentence = sentence[:start_char] + "[MASK]" + sentence[end_char:]
    
    encoded = tokenizer(masked_sentence, max_length=CONFIG['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(device)
    input_ids = encoded['input_ids'][0]
    
    mask_token_id = tokenizer.mask_token_id
    mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_indices) == 0:
        return 1.0
    mask_idx = mask_indices[0].item()
    
    target_tokens = tokenizer.tokenize(target_word)
    if not target_tokens:
        return 1.0
    target_token_id = tokenizer.convert_tokens_to_ids(target_tokens[0])
    
    # Run model (MLM predictions)
    bert_model.eval()
    with torch.no_grad():
        if hasattr(bert_model, 'mlm_head'):
            outputs = bert_model.bert(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            mask_hidden = outputs.last_hidden_state[0, mask_idx, :].unsqueeze(0)
            logits = bert_model.mlm_head(mask_hidden)[0]
        else:
            outputs = bert_model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            logits = outputs.logits[0, mask_idx]
            
        probs = torch.softmax(logits, dim=-1)
        prob = probs[target_token_id].item()
        
    surprisal = -math.log10(max(1e-9, prob))
    norm_surprisal = min(1.0, max(0.0, surprisal / 9.0))
    return norm_surprisal

# ---------------------------------------------------------
# Helper: Combined 4-signal complexity score
# ---------------------------------------------------------
def compute_combined_complexity(word: str, lemma: str, sentence: str, start_char: int, end_char: int, bert_model: nn.Module, tokenizer: BertTokenizer, device: torch.device) -> Dict[str, float]:
    # 1. BERT Surprisal (50%)
    try:
        norm_surprisal = compute_bert_surprisal(sentence, start_char, end_char, word, bert_model, tokenizer, device)
    except Exception:
        norm_surprisal = 1.0
        
    # 2. Frequency Backup (20%)
    zipf_val = zipf_frequency(word.lower(), 'en')
    norm_frequency = 1.0 - min(1.0, zipf_val / 8.0)
    
    # 3. Word Familiarity (15%)
    w_lower = word.lower()
    lemma_lower = lemma.lower() if lemma else w_lower
    if w_lower in DALE_CHALL_WORDS or lemma_lower in DALE_CHALL_WORDS:
        familiarity = 0.0
    elif w_lower in OXFORD_WORDS or lemma_lower in OXFORD_WORDS:
        familiarity = 0.2
    else:
        familiarity = 1.0
        
    # 4. Morphological (15%)
    word_len = len(word)
    len_norm = min(1.0, word_len / 15.0)
    
    syllables = ComplexWordIdentifier.count_syllables(word)
    syl_norm = min(1.0, syllables / 5.0)
    
    complex_suffixes = ('ification', 'ibility', 'ability', 'ness', 'ment', 'able', 'ious', 'ance', 'ence', 'tional', 'ative')
    suffix_boost = 0.2 if w_lower.endswith(complex_suffixes) else 0.0
    
    morphological = min(1.0, 0.4 * len_norm + 0.4 * syl_norm + 0.2 * suffix_boost)
    
    # Combined score
    combined = 0.5 * norm_surprisal + 0.2 * norm_frequency + 0.15 * familiarity + 0.15 * morphological
    
    return {
        'surprisal': norm_surprisal,
        'frequency': norm_frequency,
        'familiarity': familiarity,
        'morphological': morphological,
        'combined': combined
    }

# ---------------------------------------------------------
# Stage 1: Preprocessor
# ---------------------------------------------------------
class Preprocessor:
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)
        self.skip_pos = {
            'DET', 'PRON', 'ADP', 'CCONJ', 'SCONJ',
            'PUNCT', 'SPACE', 'NUM', 'PROPN'
        }

    def process(self, sentence: str) -> List[Dict[str, Any]]:
        doc = self.nlp(sentence)
        tokens = []
        for token in doc:
            if token.pos_ in self.skip_pos:
                continue
            tokens.append({
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'index': token.i,
                'start_char': token.idx,
                'end_char': token.idx + len(token.text)
            })
        return tokens

# ---------------------------------------------------------
# Stage 2: ComplexWordIdentifier
# ---------------------------------------------------------
class ComplexWordIdentifier:
    def __init__(self, freq_threshold: float = 4.0):
        # We store threshold but mostly rely on the combined 4-signal complexity score
        self.threshold = 0.35

    @staticmethod
    def count_syllables(word: str) -> int:
        word = word.lower().strip()
        if not word:
            return 0
        vowels = "aeiouy"
        count = 0
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1
        return max(1, count)

    def identify_complex_words(self, tokens: List[Dict[str, Any]], sentence: str, bert_model: nn.Module, tokenizer: BertTokenizer, device: torch.device) -> List[Dict[str, Any]]:
        complex_tokens = []
        for t in tokens:
            word = t['text']
            lemma = t['lemma']
            start_char = t['start_char']
            end_char = t['end_char']
            
            comp_details = compute_combined_complexity(
                word, lemma, sentence, start_char, end_char,
                bert_model, tokenizer, device
            )
            score = comp_details['combined']
            
            if score >= self.threshold:
                ct = t.copy()
                ct.update({
                    'complexity_score': score,
                    'surprisal': comp_details['surprisal'],
                    'frequency': comp_details['frequency'],
                    'familiarity': comp_details['familiarity'],
                    'morphological': comp_details['morphological'],
                    'reasons': f"score({score:.2f}) [surp={comp_details['surprisal']:.2f}, freq={comp_details['frequency']:.2f}, fam={comp_details['familiarity']:.2f}, morph={comp_details['morphological']:.2f}]"
                })
                complex_tokens.append(ct)
                
        # Fallback: Pick the word with the highest combined complexity score if none met the threshold
        if not complex_tokens and tokens:
            scored_tokens = []
            for t in tokens:
                word = t['text']
                lemma = t['lemma']
                start_char = t['start_char']
                end_char = t['end_char']
                comp_details = compute_combined_complexity(
                    word, lemma, sentence, start_char, end_char,
                    bert_model, tokenizer, device
                )
                scored_tokens.append((t, comp_details))
            
            scored_tokens.sort(key=lambda x: x[1]['combined'], reverse=True)
            best_t, best_details = scored_tokens[0]
            ct = best_t.copy()
            ct.update({
                'complexity_score': best_details['combined'],
                'surprisal': best_details['surprisal'],
                'frequency': best_details['frequency'],
                'familiarity': best_details['familiarity'],
                'morphological': best_details['morphological'],
                'reasons': f"fallback({best_details['combined']:.2f}) [surp={best_details['surprisal']:.2f}, freq={best_details['frequency']:.2f}, fam={best_details['familiarity']:.2f}, morph={best_details['morphological']:.2f}]"
            })
            complex_tokens.append(ct)
            
        return complex_tokens

# ---------------------------------------------------------
# Stage 3: Word Sense Disambiguation (WSD)
# ---------------------------------------------------------
def disambiguate_word_sense(sentence: str, target_word: str, start_char: int, end_char: int, target_pos: str, bert_model: nn.Module, tokenizer: BertTokenizer, device: torch.device) -> Optional[Any]:
    pos_map = {
        'NOUN': wn.NOUN,
        'VERB': wn.VERB,
        'ADJ': wn.ADJ,
        'ADV': wn.ADV,
        'PROPN': wn.NOUN
    }
    wn_pos = pos_map.get(target_pos)
    synsets = wn.synsets(target_word.lower(), pos=wn_pos)
    if not synsets:
        synsets = wn.synsets(target_word.lower())
    if not synsets:
        return None
        
    bert_model.eval()
    with torch.no_grad():
        encoded = tokenizer(sentence, max_length=CONFIG['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(device)
        
        prefix_text = sentence[:start_char]
        prefix_tokens = tokenizer.tokenize(prefix_text)
        target_tokens = tokenizer.tokenize(target_word)
        start_idx = len(prefix_tokens) + 1
        end_idx = start_idx + len(target_tokens)
        
        if hasattr(bert_model, 'bert'):
            outputs = bert_model.bert(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
        else:
            outputs = bert_model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            
        hidden_states = outputs[0]
        seq_len = hidden_states.size(1)
        start_idx_c = min(start_idx, seq_len - 1)
        end_idx_c = min(max(end_idx, start_idx_c + 1), seq_len)
        target_embed = hidden_states[0, start_idx_c:end_idx_c].mean(dim=0)
        
    best_synset = synsets[0]
    max_similarity = -1.0
    
    for synset in synsets:
        examples = synset.examples()
        if not examples:
            examples = [f"The meaning of this word is {synset.definition()}.", f"This is a {target_word}."]
            
        sense_embeds = []
        for example in examples:
            ex_lower = example.lower()
            t_lower = target_word.lower()
            idx = ex_lower.find(t_lower)
            if idx == -1:
                idx = max(0, len(example) // 2)
            
            ex_start = idx
            ex_end = idx + len(t_lower)
            
            with torch.no_grad():
                ex_encoded = tokenizer(example, max_length=CONFIG['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(device)
                ex_prefix = example[:ex_start]
                ex_prefix_tokens = tokenizer.tokenize(ex_prefix)
                ex_target_tokens = tokenizer.tokenize(target_word)
                ex_start_idx = len(ex_prefix_tokens) + 1
                ex_end_idx = ex_start_idx + len(ex_target_tokens)
                
                if hasattr(bert_model, 'bert'):
                    ex_outputs = bert_model.bert(input_ids=ex_encoded['input_ids'], attention_mask=ex_encoded['attention_mask'])
                else:
                    ex_outputs = bert_model(input_ids=ex_encoded['input_ids'], attention_mask=ex_encoded['attention_mask'])
                    
                ex_hidden = ex_outputs[0]
                ex_seq_len = ex_hidden.size(1)
                ex_start_idx_c = min(ex_start_idx, ex_seq_len - 1)
                ex_end_idx_c = min(max(ex_end_idx, ex_start_idx_c + 1), ex_seq_len)
                ex_embed = ex_hidden[0, ex_start_idx_c:ex_end_idx_c].mean(dim=0)
                
                if not torch.isnan(ex_embed).any():
                    sense_embeds.append(ex_embed)
                    
        if sense_embeds:
            avg_sense_embed = torch.stack(sense_embeds).mean(dim=0)
            similarity = F.cosine_similarity(target_embed.unsqueeze(0), avg_sense_embed.unsqueeze(0)).item()
            if similarity > max_similarity:
                max_similarity = similarity
                best_synset = synset
                
    return best_synset

# ---------------------------------------------------------
# Stage 4: Sense-Specific Candidate Generation
# ---------------------------------------------------------
class CandidateGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.glove_name = config['glove_model']
        print(f"Loading GloVe Embeddings: {self.glove_name}...")
        
        local_path = os.path.expanduser(f"~/gensim-data/{self.glove_name}/{self.glove_name}.gz")
        if os.path.exists(local_path):
            try:
                self.glove = KeyedVectors.load_word2vec_format(local_path, binary=False, no_header=False)
                print("GloVe loaded successfully from local cache.")
            except Exception as e:
                print(f"Failed loading offline GloVe: {e}. Downloading via online API...")
                self.glove = api.load(self.glove_name)
        else:
            self.glove = api.load(self.glove_name)

    def get_wordnet_candidates(self, chosen_synset: Any) -> Set[str]:
        if not chosen_synset:
            return set()
        candidates = set()
        for lemma in chosen_synset.lemmas():
            clean_name = lemma.name().replace('_', ' ').replace('-', ' ').lower()
            candidates.add(clean_name)
        return candidates

    def get_glove_candidates(self, word: str, top_n: int = 10) -> Set[str]:
        if self.glove is None:
            return set()
        try:
            similar_words = self.glove.most_similar(word, topn=top_n)
            return {w[0].lower() for w in similar_words}
        except KeyError:
            return set()

# Grammatical particles to exclude
STOP_WORDS = {
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
    'although', 'because', 'since', 'unless', 'while', 'whereas'
}

# ---------------------------------------------------------
# Dataset class parsing BenchLS
# ---------------------------------------------------------
class LexicalSimplificationDataset(Dataset):
    def __init__(self, config: Dict[str, Any], tokenizer: BertTokenizer, data_path: str):
        self.config = config
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.nlp = spacy.load("en_core_web_sm")
        self.samples = self._load_or_generate_dataset()

    def _load_or_generate_dataset(self) -> List[Dict[str, Any]]:
        benchls_txt = os.path.join(self.data_path, "BenchLS.txt")
        if os.path.exists(benchls_txt):
            print(f"Parsing BenchLS dataset from {benchls_txt}...")
            return self._parse_benchls_file(benchls_txt)
        else:
            raise FileNotFoundError(f"BenchLS.txt not found in {self.data_path}")

    def _parse_benchls_file(self, file_path: str) -> List[Dict[str, Any]]:
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) < 4: continue
                sentence = parts[0]
                target = parts[1]
                try:
                    position = int(parts[2])
                except ValueError:
                    continue
                words = sentence.split(' ')
                if position >= len(words):
                    match = re.search(r'\b' + re.escape(target) + r'\b', sentence, re.IGNORECASE)
                    if not match: continue
                    start_char = match.start()
                    end_char = match.end()
                else:
                    prefix = " ".join(words[:position])
                    start_char = len(prefix) + 1 if position > 0 else 0
                    end_char = start_char + len(target)
                
                target_pos = 'NOUN'
                try:
                    doc = self.nlp(sentence)
                    for token in doc:
                        if token.idx == start_char or (token.idx <= start_char and token.idx + len(token.text) >= end_char):
                            target_pos = token.pos_
                            break
                except Exception:
                    pass
                    
                candidates_parts = parts[3:]
                parsed_cands = []
                for item in candidates_parts:
                    if ':' in item:
                        rank_str, cand = item.split(':', 1)
                        try:
                            rank = int(rank_str)
                            parsed_cands.append((cand, rank))
                        except ValueError:
                            try:
                                cand_str, r_str = item.rsplit(':', 1)
                                rank = int(r_str)
                                parsed_cands.append((cand_str, rank))
                            except ValueError:
                                continue
                if not parsed_cands: continue
                for cand_word, rank in parsed_cands:
                    label = 1.0 / float(rank)
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # We use a PrecomputedDataset wrapper during training, so this is fallback only
        return self.samples[idx]

# ---------------------------------------------------------
# Stage 5: Neural Contextual Ranker Model
# ---------------------------------------------------------
class LexicalSimplificationModel(nn.Module):
    def __init__(self, config: Dict[str, Any], vocab_size: int):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['bert_model'])
        
        self.fine_tune_bert = config.get('fine_tune_bert', False)
        if not self.fine_tune_bert:
            print("Freezing BERT encoder weights for fast CPU training...")
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.mlm_head = nn.Linear(768, vocab_size)
        print("Initializing MLM head with pretrained BERT weights...")
        try:
            pretrained_mlm = BertForMaskedLM.from_pretrained(config['bert_model'])
            with torch.no_grad():
                self.mlm_head.weight.copy_(pretrained_mlm.cls.predictions.decoder.weight)
                self.mlm_head.bias.copy_(pretrained_mlm.cls.predictions.bias)
            print("MLM head weights successfully loaded.")
            del pretrained_mlm
        except Exception as e:
            print(f"Could not load pretrained MLM weights: {e}. Random init.")
            
        self.cosine = nn.CosineSimilarity(dim=1)
        
        # 1. CLS Context projection (768 -> 16)
        self.context_proj = nn.Linear(768, 16)
        
        # 2. MLM probability projection (1 -> 4)
        self.mlm_proj = nn.Linear(1, 4)
        
        # Combined features ranker: context_proj (16) + mlm_proj (4) + semantic cosine (1) + simplicity delta (1) = 22
        self.ranker_net = nn.Sequential(
            nn.Linear(16 + 4 + 1 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, 
                context_embed: torch.Tensor,
                mlm_prob: torch.Tensor,
                semantic_similarity: torch.Tensor,
                simplicity_delta: torch.Tensor) -> torch.Tensor:
        
        context_features = self.context_proj(context_embed)
        mlm_features = self.mlm_proj(mlm_prob)
        
        features = torch.cat([context_features, mlm_features, semantic_similarity, simplicity_delta], dim=-1)
        score = self.ranker_net(features)
        return score

# ---------------------------------------------------------
# Batched Precomputation of Features
# ---------------------------------------------------------
def precompute_all_features(samples: List[Dict[str, Any]], model: nn.Module, tokenizer: BertTokenizer, device: torch.device) -> List[Dict[str, torch.Tensor]]:
    print(f"Precomputing 6-stage pipeline features for {len(samples)} samples...")
    cache = []
    
    # Group by unique sentence/target to minimize BERT passes
    grouped = {}
    for i, s in enumerate(samples):
        key = (s['sentence'], s['target_word'], s['start_char'], s['end_char'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append((i, s))
        
    model.eval()
    with torch.no_grad():
        for (sentence, target_word, start_char, end_char), group_items in tqdm(grouped.items(), desc="Precomputing"):
            # 1. Encode original sentence
            encoded = tokenizer(sentence, max_length=CONFIG['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(device)
            outputs = model.bert(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            context_embed = outputs.last_hidden_state[0, 0, :].cpu() # CLS
            
            # Original word contextual embedding
            prefix_text = sentence[:start_char]
            prefix_tokens = tokenizer.tokenize(prefix_text)
            target_tokens = tokenizer.tokenize(target_word)
            start_idx = len(prefix_tokens) + 1
            end_idx = start_idx + len(target_tokens)
            
            seq_len = outputs.last_hidden_state.size(1)
            start_idx_c = min(start_idx, seq_len - 1)
            end_idx_c = min(max(end_idx, start_idx_c + 1), seq_len)
            orig_contextual_embed = outputs.last_hidden_state[0, start_idx_c:end_idx_c].mean(dim=0).cpu()
            
            # 2. Masked sentence predictions
            masked_sentence = sentence[:start_char] + "[MASK]" + sentence[end_char:]
            masked_encoded = tokenizer(masked_sentence, max_length=CONFIG['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(device)
            masked_outputs = model.bert(input_ids=masked_encoded['input_ids'], attention_mask=masked_encoded['attention_mask'])
            
            mask_token_id = tokenizer.mask_token_id
            mask_indices = (masked_encoded['input_ids'][0] == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_indices) > 0:
                mask_idx = mask_indices[0].item()
            else:
                mask_idx = min(start_idx, CONFIG['max_length'] - 1)
                
            mask_hidden = masked_outputs.last_hidden_state[0, mask_idx, :]
            mlm_logits = model.mlm_head(mask_hidden.unsqueeze(0)).squeeze(0)
            mlm_probs = torch.softmax(mlm_logits, dim=-1).cpu()
            
            # Target complexity
            target_token_id = tokenizer.convert_tokens_to_ids(target_tokens[0]) if target_tokens else tokenizer.unk_token_id
            target_mlm_prob = mlm_probs[target_token_id].item()
            target_surprisal = -math.log10(max(1e-9, target_mlm_prob))
            target_surprisal_norm = min(1.0, max(0.0, target_surprisal / 9.0))
            
            zipf_val = zipf_frequency(target_word.lower(), 'en')
            target_freq_norm = 1.0 - min(1.0, zipf_val / 8.0)
            
            target_w_lower = target_word.lower()
            if target_w_lower in DALE_CHALL_WORDS:
                target_fam = 0.0
            elif target_w_lower in OXFORD_WORDS:
                target_fam = 0.2
            else:
                target_fam = 1.0
                
            target_len_norm = min(1.0, len(target_word) / 15.0)
            target_syl = ComplexWordIdentifier.count_syllables(target_word)
            target_syl_norm = min(1.0, target_syl / 5.0)
            complex_suffixes = ('ification', 'ibility', 'ability', 'ness', 'ment', 'able', 'ious', 'ance', 'ence', 'tional', 'ative')
            target_suffix_boost = 0.2 if target_w_lower.endswith(complex_suffixes) else 0.0
            target_morph = min(1.0, 0.4 * target_len_norm + 0.4 * target_syl_norm + 0.2 * target_suffix_boost)
            
            target_complexity = 0.5 * target_surprisal_norm + 0.2 * target_freq_norm + 0.15 * target_fam + 0.15 * target_morph
            
            # Process candidates
            for idx, s in group_items:
                candidate_word = s['candidate']
                cand_lower = candidate_word.lower()
                
                cand_tokens = tokenizer.tokenize(candidate_word)
                cand_token_id = tokenizer.convert_tokens_to_ids(cand_tokens[0]) if cand_tokens else tokenizer.unk_token_id
                
                mlm_prob_val = mlm_probs[cand_token_id].item()
                
                # Candidate sentence contextual embedding
                cand_sentence = sentence[:start_char] + candidate_word + sentence[end_char:]
                cand_encoded = tokenizer(cand_sentence, max_length=CONFIG['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(device)
                cand_outputs = model.bert(input_ids=cand_encoded['input_ids'], attention_mask=cand_encoded['attention_mask'])
                
                cand_prefix_tokens = tokenizer.tokenize(sentence[:start_char])
                cand_word_tokens = tokenizer.tokenize(candidate_word)
                cand_start_idx = len(cand_prefix_tokens) + 1
                cand_end_idx = cand_start_idx + len(cand_word_tokens)
                
                cand_seq_len = cand_outputs.last_hidden_state.size(1)
                cand_start_idx_c = min(cand_start_idx, cand_seq_len - 1)
                cand_end_idx_c = min(max(cand_end_idx, cand_start_idx_c + 1), cand_seq_len)
                cand_contextual_embed = cand_outputs.last_hidden_state[0, cand_start_idx_c:cand_end_idx_c].mean(dim=0).cpu()
                
                cosine_sim = F.cosine_similarity(orig_contextual_embed.unsqueeze(0), cand_contextual_embed.unsqueeze(0)).item()
                
                # Candidate complexity
                cand_mlm_prob = mlm_probs[cand_token_id].item()
                cand_surprisal = -math.log10(max(1e-9, cand_mlm_prob))
                cand_surprisal_norm = min(1.0, max(0.0, cand_surprisal / 9.0))
                
                cand_zipf = zipf_frequency(cand_lower, 'en')
                cand_freq_norm = 1.0 - min(1.0, cand_zipf / 8.0)
                
                if cand_lower in DALE_CHALL_WORDS:
                    cand_fam = 0.0
                elif cand_lower in OXFORD_WORDS:
                    cand_fam = 0.2
                else:
                    cand_fam = 1.0
                    
                cand_len_norm = min(1.0, len(candidate_word) / 15.0)
                cand_syl = ComplexWordIdentifier.count_syllables(candidate_word)
                cand_syl_norm = min(1.0, cand_syl / 5.0)
                cand_suffix_boost = 0.2 if cand_lower.endswith(complex_suffixes) else 0.0
                cand_morph = min(1.0, 0.4 * cand_len_norm + 0.4 * cand_syl_norm + 0.2 * cand_suffix_boost)
                
                cand_complexity = 0.5 * cand_surprisal_norm + 0.2 * cand_freq_norm + 0.15 * cand_fam + 0.15 * cand_morph
                
                simplicity_delta = target_complexity - cand_complexity
                
                cache.append({
                    'index': idx,
                    'context_embed': context_embed,
                    'mlm_prob': torch.tensor([mlm_prob_val], dtype=torch.float32),
                    'semantic_similarity': torch.tensor([cosine_sim], dtype=torch.float32),
                    'simplicity_delta': torch.tensor([simplicity_delta], dtype=torch.float32),
                    'label': torch.tensor(s['label'], dtype=torch.float32)
                })
                
    cache.sort(key=lambda x: x['index'])
    return cache

# Precomputed Dataset wrapper
class PrecomputedDataset(Dataset):
    def __init__(self, cached_features):
        self.features = cached_features
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx]

# ---------------------------------------------------------
# Training loop functions
# ---------------------------------------------------------
def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: Any, device: torch.device, grad_clip: float) -> float:
    model.train()
    total_loss = 0.0
    mse_criterion = nn.MSELoss()
    
    for batch in tqdm(dataloader, desc="Training Batches", leave=False):
        context_embed = batch['context_embed'].to(device)
        mlm_prob = batch['mlm_prob'].to(device)
        semantic_similarity = batch['semantic_similarity'].to(device)
        simplicity_delta = batch['simplicity_delta'].to(device)
        labels = batch['label'].to(device).unsqueeze(-1)
        
        optimizer.zero_grad()
        scores = model(context_embed, mlm_prob, semantic_similarity, simplicity_delta)
        loss = mse_criterion(scores, labels)
        
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * context_embed.size(0)
        
    return total_loss / len(dataloader.dataset)

def validate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    mse_criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in dataloader:
            context_embed = batch['context_embed'].to(device)
            mlm_prob = batch['mlm_prob'].to(device)
            semantic_similarity = batch['semantic_similarity'].to(device)
            simplicity_delta = batch['simplicity_delta'].to(device)
            labels = batch['label'].to(device).unsqueeze(-1)
            
            scores = model(context_embed, mlm_prob, semantic_similarity, simplicity_delta)
            loss = mse_criterion(scores, labels)
            
            total_loss += loss.item() * context_embed.size(0)
            
    return total_loss / len(dataloader.dataset)

def evaluate_model(model: nn.Module, val_samples: List[Dict[str, Any]], tokenizer: BertTokenizer, device: torch.device):
    return evaluate_model_static(model, val_samples, tokenizer, device, DALE_CHALL_WORDS, OXFORD_WORDS)

def evaluate_model_static(model: nn.Module, val_samples: List[Dict[str, Any]], tokenizer: BertTokenizer, device: torch.device, dale_chall_words: Set[str], oxford_words: Set[str]):
    model.eval()
    
    grouped = {}
    for s in val_samples:
        key = (s['sentence'], s['target_word'], s['start_char'], s['end_char'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(s)
        
    total_instances = len(grouped)
    successful_p1 = 0
    successful_acc = 0
    words_changed = 0
    
    with torch.no_grad():
        for (sentence, target_word, start_char, end_char), candidates_list in grouped.items():
            encoded = tokenizer(sentence, max_length=CONFIG['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(device)
            outputs = model.bert(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            context_embed = outputs.last_hidden_state[0, 0, :].cpu()
            
            prefix_text = sentence[:start_char]
            prefix_tokens = tokenizer.tokenize(prefix_text)
            target_tokens = tokenizer.tokenize(target_word)
            start_idx = len(prefix_tokens) + 1
            end_idx = start_idx + len(target_tokens)
            
            seq_len = outputs.last_hidden_state.size(1)
            start_idx_c = min(start_idx, seq_len - 1)
            end_idx_c = min(max(end_idx, start_idx_c + 1), seq_len)
            orig_contextual_embed = outputs.last_hidden_state[0, start_idx_c:end_idx_c].mean(dim=0).cpu()
            
            masked_sentence = sentence[:start_char] + "[MASK]" + sentence[end_char:]
            masked_encoded = tokenizer(masked_sentence, max_length=CONFIG['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(device)
            masked_outputs = model.bert(input_ids=masked_encoded['input_ids'], attention_mask=masked_encoded['attention_mask'])
            
            mask_token_id = tokenizer.mask_token_id
            mask_indices = (masked_encoded['input_ids'][0] == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_indices) > 0:
                mask_idx = mask_indices[0].item()
            else:
                mask_idx = min(start_idx, CONFIG['max_length'] - 1)
                
            mask_hidden = masked_outputs.last_hidden_state[0, mask_idx, :]
            mlm_logits = model.mlm_head(mask_hidden.unsqueeze(0)).squeeze(0)
            mlm_probs = torch.softmax(mlm_logits, dim=-1).cpu()
            
            target_token_id = tokenizer.convert_tokens_to_ids(target_tokens[0]) if target_tokens else tokenizer.unk_token_id
            target_mlm_prob = mlm_probs[target_token_id].item()
            target_surprisal = -math.log10(max(1e-9, target_mlm_prob))
            target_surprisal_norm = min(1.0, max(0.0, target_surprisal / 9.0))
            
            zipf_val = zipf_frequency(target_word.lower(), 'en')
            target_freq_norm = 1.0 - min(1.0, zipf_val / 8.0)
            
            target_w_lower = target_word.lower()
            if target_w_lower in dale_chall_words:
                target_fam = 0.0
            elif target_w_lower in oxford_words:
                target_fam = 0.2
            else:
                target_fam = 1.0
                
            target_len_norm = min(1.0, len(target_word) / 15.0)
            target_syl = ComplexWordIdentifier.count_syllables(target_word)
            target_syl_norm = min(1.0, target_syl / 5.0)
            complex_suffixes = ('ification', 'ibility', 'ability', 'ness', 'ment', 'able', 'ious', 'ance', 'ence', 'tional', 'ative')
            target_suffix_boost = 0.2 if target_w_lower.endswith(complex_suffixes) else 0.0
            target_morph = min(1.0, 0.4 * target_len_norm + 0.4 * target_syl_norm + 0.2 * target_suffix_boost)
            
            target_complexity = 0.5 * target_surprisal_norm + 0.2 * target_freq_norm + 0.15 * target_fam + 0.15 * target_morph
            
            scored_candidates = []
            gold_best_words = set()
            gold_all_words = set()
            
            best_label = -1.0
            for s in candidates_list:
                cand = s['candidate'].lower()
                gold_all_words.add(cand)
                if s['label'] > best_label:
                    best_label = s['label']
                    
            for s in candidates_list:
                if abs(s['label'] - best_label) < 1e-5:
                    gold_best_words.add(s['candidate'].lower())
                    
            for s in candidates_list:
                candidate_word = s['candidate']
                cand_lower = candidate_word.lower()
                
                cand_tokens = tokenizer.tokenize(candidate_word)
                cand_token_id = tokenizer.convert_tokens_to_ids(cand_tokens[0]) if cand_tokens else tokenizer.unk_token_id
                
                mlm_prob_val = mlm_probs[cand_token_id].item()
                
                # Contextual embedding candidate sentence
                cand_sentence = sentence[:start_char] + candidate_word + sentence[end_char:]
                cand_encoded = tokenizer(cand_sentence, max_length=CONFIG['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(device)
                cand_outputs = model.bert(input_ids=cand_encoded['input_ids'], attention_mask=cand_encoded['attention_mask'])
                
                cand_prefix_tokens = tokenizer.tokenize(sentence[:start_char])
                cand_word_tokens = tokenizer.tokenize(candidate_word)
                cand_start_idx = len(cand_prefix_tokens) + 1
                cand_end_idx = cand_start_idx + len(cand_word_tokens)
                
                cand_seq_len = cand_outputs.last_hidden_state.size(1)
                cand_start_idx_c = min(cand_start_idx, cand_seq_len - 1)
                cand_end_idx_c = min(max(cand_end_idx, cand_start_idx_c + 1), cand_seq_len)
                cand_contextual_embed = cand_outputs.last_hidden_state[0, cand_start_idx_c:cand_end_idx_c].mean(dim=0).cpu()
                
                cosine_sim = F.cosine_similarity(orig_contextual_embed.unsqueeze(0), cand_contextual_embed.unsqueeze(0)).item()
                
                # Candidate complexity
                cand_mlm_prob = mlm_probs[cand_token_id].item()
                cand_surprisal = -math.log10(max(1e-9, cand_mlm_prob))
                cand_surprisal_norm = min(1.0, max(0.0, cand_surprisal / 9.0))
                
                cand_zipf = zipf_frequency(cand_lower, 'en')
                cand_freq_norm = 1.0 - min(1.0, cand_zipf / 8.0)
                
                if cand_lower in dale_chall_words:
                    cand_fam = 0.0
                elif cand_lower in oxford_words:
                    cand_fam = 0.2
                else:
                    cand_fam = 1.0
                    
                cand_len_norm = min(1.0, len(candidate_word) / 15.0)
                cand_syl = ComplexWordIdentifier.count_syllables(candidate_word)
                cand_syl_norm = min(1.0, cand_syl / 5.0)
                cand_suffix_boost = 0.2 if cand_lower.endswith(complex_suffixes) else 0.0
                cand_morph = min(1.0, 0.4 * cand_len_norm + 0.4 * cand_syl_norm + 0.2 * cand_suffix_boost)
                
                cand_complexity = 0.5 * cand_surprisal_norm + 0.2 * cand_freq_norm + 0.15 * cand_fam + 0.15 * cand_morph
                
                simplicity_delta = target_complexity - cand_complexity
                
                feat_context = context_embed.unsqueeze(0).to(device)
                feat_mlm = torch.tensor([[mlm_prob_val]], dtype=torch.float32).to(device)
                feat_sim = torch.tensor([[cosine_sim]], dtype=torch.float32).to(device)
                feat_delta = torch.tensor([[simplicity_delta]], dtype=torch.float32).to(device)
                
                pred_score = model(feat_context, feat_mlm, feat_sim, feat_delta).item()
                scored_candidates.append((cand_lower, pred_score))
                
            if scored_candidates:
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                top_pred_word = scored_candidates[0][0]
                if top_pred_word in gold_best_words:
                    successful_p1 += 1
                if top_pred_word in gold_all_words:
                    successful_acc += 1
                words_changed += 1
                
    p1_score = (successful_p1 / total_instances) * 100 if total_instances > 0 else 0.0
    acc_score = (successful_acc / total_instances) * 100 if total_instances > 0 else 0.0
    changed_metric = (words_changed / total_instances) * 100 if total_instances > 0 else 0.0
    
    print("\n" + "="*45)
    print("           EVALUATION METRICS TABLE")
    print("="*45)
    print(f"  Metric Name               | Value")
    print("-"*45)
    print(f"  Precision@1 Score         | {p1_score:.2f}%")
    print(f"  Accuracy (Standard)       | {acc_score:.2f}%")
    print(f"  Changed Words Metric      | {changed_metric:.2f}%")
    print(f"  Total Sample Instances    | {total_instances}")
    print("="*45 + "\n")
    return {'Precision@1': p1_score, 'Accuracy': acc_score, 'Changed_Metric': changed_metric}

def inflect_candidate(target_word: str, target_pos: str, candidate_word: str) -> str:
    """
    Adjusts the grammatical form (inflection) of candidate_word to match target_word.
    Handles singular/plural nouns and verb tenses (-ing, -ed, -s).
    """
    target_pos = target_pos.upper()
    cand_lower = candidate_word.lower()
    
    # 1. Singular/Plural Nouns
    if target_pos == 'NOUN' or target_pos == 'PROPN':
        if target_word.endswith('s') and not target_word.endswith('ss'):
            # Target is likely plural, pluralize candidate
            if not cand_lower.endswith('s'):
                if cand_lower.endswith(('ch', 'sh', 'x', 'z', 'o')):
                    return cand_lower + 'es'
                elif cand_lower.endswith('y') and len(cand_lower) > 1 and cand_lower[-2] not in 'aeiou':
                    return cand_lower[:-1] + 'ies'
                else:
                    return cand_lower + 's'
        elif not target_word.endswith('s') or target_word.endswith('ss'):
            # Target is singular, ensure candidate is singular
            if cand_lower.endswith('s') and not cand_lower.endswith('ss'):
                if cand_lower.endswith('ies'):
                    return cand_lower[:-3] + 'y'
                elif cand_lower.endswith('es') and cand_lower[:-2].endswith(('ch', 'sh', 'x', 'z', 'o')):
                    return cand_lower[:-2]
                else:
                    return cand_lower[:-1]
                    
    # 2. Verbs (-ing, -ed, -s)
    elif target_pos == 'VERB':
        if target_word.endswith('ing'):
            if not cand_lower.endswith('ing'):
                if cand_lower.endswith('e') and not cand_lower.endswith(('ee', 'ye', 'oe')):
                    return cand_lower[:-1] + 'ing'
                else:
                    return cand_lower + 'ing'
        elif target_word.endswith('ed'):
            if not cand_lower.endswith('ed'):
                if cand_lower.endswith('e'):
                    return cand_lower + 'd'
                elif cand_lower.endswith('y') and len(cand_lower) > 1 and cand_lower[-2] not in 'aeiou':
                    return cand_lower[:-1] + 'ied'
                else:
                    return cand_lower + 'ed'
        elif target_word.endswith('s') and not target_word.endswith('ss'):
            if not cand_lower.endswith('s'):
                if cand_lower.endswith(('ch', 'sh', 'x', 'z', 'o')):
                    return cand_lower + 'es'
                elif cand_lower.endswith('y') and len(cand_lower) > 1 and cand_lower[-2] not in 'aeiou':
                    return cand_lower[:-1] + 'ies'
                else:
                    return cand_lower + 's'
                    
    return cand_lower

def are_semantically_related(chosen_sense: Optional[Any], target_word: str, cand: str, pos_tag: str) -> bool:
    pos_map = {
        'NOUN': wn.NOUN,
        'VERB': wn.VERB,
        'ADJ': wn.ADJ,
        'ADV': wn.ADV,
        'PROPN': wn.NOUN
    }
    wn_pos = pos_map.get(pos_tag)
    if not wn_pos:
        return True
        
    c_s = wn.synsets(cand.lower(), pos=wn_pos)
    if not c_s:
        return False
        
    # If WSD succeeded, we check relationship to the chosen_sense
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

# ---------------------------------------------------------
# Stage-by-Stage Inference simplifies pipeline
# ---------------------------------------------------------
class LexicalSimplifier:
    def __init__(self, config: Dict[str, Any], model_path: str, preprocessor: Preprocessor, cwi: ComplexWordIdentifier, candidate_generator: CandidateGenerator, tokenizer: BertTokenizer, device: torch.device):
        self.config = config
        self.device = device
        self.preprocessor = preprocessor
        self.cwi = cwi
        self.generator = candidate_generator
        self.tokenizer = tokenizer
        
        self.model = LexicalSimplificationModel(config, tokenizer.vocab_size)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded trained ranker weights from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load weights from {model_path} ({e}). Using initialized weights.")
        else:
            print("Warning: Checkpoint not found. Using initialized weights.")
        self.model.to(device)
        self.model.eval()

    def simplify(self, sentence: str) -> str:
        print("\n" + "="*60)
        print(f"INPUT SENTENCE : {sentence}")
        print("="*60)
        
        # Stage 1: Preprocessing
        tokens = self.preprocessor.process(sentence)
        tokens_repr = [f"{t['text']}({t['pos']})" for t in tokens]
        print(f"\nStage 1 - Preprocessing:")
        print(f"  Tokens: [{', '.join(tokens_repr)}]")
        
        # Stage 2: Context-Aware CWI
        complex_tokens = self.cwi.identify_complex_words(tokens, sentence, self.model, self.tokenizer, self.device)
        if not complex_tokens:
            print("\nStage 2 - Complex Words Found: None.")
            return sentence
            
        print("\nStage 2 - Complex Words Found:")
        # Display prioritized by complexity score descending
        prioritized_tokens = sorted(complex_tokens, key=lambda x: x['complexity_score'], reverse=True)
        for cw in prioritized_tokens:
            print(f"  [Complex Word] '{cw['text']}' | Score: {cw['complexity_score']:.3f} | Reasons: {cw['reasons']}")
            
        # Iterate in correct grammatical order (sorted by original start_char / index)
        grammatical_tokens = sorted(complex_tokens, key=lambda x: x['start_char'])
        
        current_sentence = sentence
        offset_shift = 0
        
        print(f"\nProceeding with sequential substitution of {len(grammatical_tokens)} complex word(s)...")
        
        for idx_w, target in enumerate(grammatical_tokens):
            word = target['text']
            pos = target['pos']
            orig_start = target['start_char']
            orig_end = target['end_char']
            
            # Adjusted positions
            start_char = orig_start + offset_shift
            end_char = orig_end + offset_shift
            
            # Sanity check: verify the target word is indeed present at the adjusted coordinates
            actual_word_in_sentence = current_sentence[start_char:end_char]
            if actual_word_in_sentence.lower() != word.lower():
                # Fallback to search dynamically if offsets mismatched
                dynamic_start = current_sentence.find(word)
                if dynamic_start == -1:
                    dynamic_start = current_sentence.lower().find(word.lower())
                if dynamic_start == -1:
                    print(f"\nSkipping '{word}' (could not locate in current sentence).")
                    continue
                start_char = dynamic_start
                end_char = start_char + len(word)
                
            print(f"\n--- Stage 3, 4, 5, 6 for Word {idx_w + 1}/{len(grammatical_tokens)}: '{word}' ({pos}) ---")
            
            # Stage 3: WSD
            chosen_sense = disambiguate_word_sense(current_sentence, word, start_char, end_char, pos, self.model, self.tokenizer, self.device)
            print(f"Stage 3 - Word Sense Disambiguation:")
            if chosen_sense:
                print(f"  Chosen Sense  : {chosen_sense.name()}")
                print(f"  Definition    : {chosen_sense.definition()}")
            else:
                print("  Chosen Sense  : None (WordNet lookup failed)")
                
            # Stage 4: Sense-Specific Candidate Generation & Filtering
            wn_cands = self.generator.get_wordnet_candidates(chosen_sense)
            glove_cands = self.generator.get_glove_candidates(word.lower(), top_n=30)
            
            # BERT MLM predictions at the target position in the current sentence
            masked_sentence = current_sentence[:start_char] + "[MASK]" + current_sentence[end_char:]
            encoded = self.tokenizer(current_sentence, max_length=self.config['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(self.device)
            masked_encoded = self.tokenizer(masked_sentence, max_length=self.config['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(self.device)
            
            mask_token_id = self.tokenizer.mask_token_id
            mask_indices = (masked_encoded['input_ids'][0] == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_indices) > 0:
                mask_idx = mask_indices[0].item()
            else:
                mask_idx = min(len(self.tokenizer.tokenize(current_sentence[:start_char])) + 1, self.config['max_length'] - 1)
                
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.bert(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
                context_embed = outputs.last_hidden_state[0, 0, :].cpu()
                
                prefix_tokens = self.tokenizer.tokenize(current_sentence[:start_char])
                target_tokens = self.tokenizer.tokenize(word)
                start_idx_c = min(len(prefix_tokens) + 1, outputs.last_hidden_state.size(1) - 1)
                end_idx_c = min(start_idx_c + len(target_tokens), outputs.last_hidden_state.size(1))
                orig_contextual_embed = outputs.last_hidden_state[0, start_idx_c:end_idx_c].mean(dim=0).cpu()
                
                masked_outputs = self.model.bert(input_ids=masked_encoded['input_ids'], attention_mask=masked_encoded['attention_mask'])
                mask_hidden = masked_outputs.last_hidden_state[0, mask_idx, :]
                mlm_logits = self.model.mlm_head(mask_hidden.unsqueeze(0)).squeeze(0)
                mlm_probs = torch.softmax(mlm_logits, dim=-1).cpu()
                
            top_mlm_indices = torch.topk(mlm_probs, k=25).indices.tolist()
            mlm_cands = set()
            for idx in top_mlm_indices:
                tok = self.tokenizer.decode([idx]).strip().lower()
                if tok and not tok.startswith("##") and tok.isalpha():
                    mlm_cands.add(tok)
                    
            # Merge all sources
            all_candidates = wn_cands.union(glove_cands).union(mlm_cands)
            print(f"Stage 4 - Candidate Generation (Raw Count: {len(all_candidates)}):")
            
            # Calculate target word's complexity in context
            target_token_id = self.tokenizer.convert_tokens_to_ids(target_tokens[0]) if target_tokens else self.tokenizer.unk_token_id
            target_mlm_prob = mlm_probs[target_token_id].item()
            target_surprisal = -math.log10(max(1e-9, target_mlm_prob))
            target_surprisal_norm = min(1.0, max(0.0, target_surprisal / 9.0))
            target_freq_norm = 1.0 - min(1.0, zipf_frequency(word.lower(), 'en') / 8.0)
            target_fam = 0.0 if word.lower() in DALE_CHALL_WORDS else (0.2 if word.lower() in OXFORD_WORDS else 1.0)
            target_len_norm = min(1.0, len(word) / 15.0)
            target_syl = ComplexWordIdentifier.count_syllables(word)
            target_syl_norm = min(1.0, target_syl / 5.0)
            complex_suffixes = ('ification', 'ibility', 'ability', 'ness', 'ment', 'able', 'ious', 'ance', 'ence', 'tional', 'ative')
            target_suffix_boost = 0.2 if word.lower().endswith(complex_suffixes) else 0.0
            target_morph = min(1.0, 0.4 * target_len_norm + 0.4 * target_syl_norm + 0.2 * target_suffix_boost)
            target_complexity = 0.5 * target_surprisal_norm + 0.2 * target_freq_norm + 0.15 * target_fam + 0.15 * target_morph
            
            # Filter candidates
            filtered_candidates = []
            for cand in all_candidates:
                cand_lower = cand.lower()
                if cand_lower == word.lower() or cand_lower in STOP_WORDS or cand_lower in self.preprocessor.nlp.Defaults.stop_words:
                    continue
                if ' ' in cand_lower or '-' in cand_lower or '_' in cand_lower:
                    continue
                if not cand_lower.isalpha():
                    continue
                    
                # POS and Semantic Gate Check
                if not are_semantically_related(chosen_sense, word, cand_lower, pos):
                    continue
                    
                # Candidate complexity
                cand_tokens = self.tokenizer.tokenize(cand_lower)
                if not cand_tokens:
                    continue
                cand_token_id = self.tokenizer.convert_tokens_to_ids(cand_tokens[0])
                cand_mlm_prob = mlm_probs[cand_token_id].item()
                cand_surprisal = -math.log10(max(1e-9, cand_mlm_prob))
                cand_surprisal_norm = min(1.0, max(0.0, cand_surprisal / 9.0))
                cand_freq_norm = 1.0 - min(1.0, zipf_frequency(cand_lower, 'en') / 8.0)
                cand_fam = 0.0 if cand_lower in DALE_CHALL_WORDS else (0.2 if cand_lower in OXFORD_WORDS else 1.0)
                cand_len_norm = min(1.0, len(cand_lower) / 15.0)
                cand_syl = ComplexWordIdentifier.count_syllables(cand_lower)
                cand_syl_norm = min(1.0, cand_syl / 5.0)
                cand_suffix_boost = 0.2 if cand_lower.endswith(complex_suffixes) else 0.0
                cand_morph = min(1.0, 0.4 * cand_len_norm + 0.4 * cand_syl_norm + 0.2 * cand_suffix_boost)
                cand_complexity = 0.5 * cand_surprisal_norm + 0.2 * cand_freq_norm + 0.15 * cand_fam + 0.15 * cand_morph
                
                # Enforce simplicity constraint
                if cand_complexity >= target_complexity:
                    continue
                    
                filtered_candidates.append({
                    'word': cand_lower,
                    'mlm_prob': cand_mlm_prob,
                    'complexity': cand_complexity,
                    'simplicity_delta': target_complexity - cand_complexity
                })
                
            print(f"  Filtered Candidates (Strictly Simpler): {[c['word'] for c in filtered_candidates[:15]]} ... ({len(filtered_candidates)} total)")
            if not filtered_candidates:
                print("  No simpler candidates found. Skipping replacement.")
                continue
                
            # Stage 5: Contextual Neural Ranking
            print(f"Stage 5 - Contextual Neural Ranking:")
            print(f"  Candidate  | MLM Prob | Context Cos | Sim Delta | FINAL SCORE")
            print("  " + "-"*56)
            
            scored_candidates = []
            for c in filtered_candidates:
                cand_word = c['word']
                
                # Compute candidate contextual embedding
                cand_sentence = current_sentence[:start_char] + cand_word + current_sentence[end_char:]
                with torch.no_grad():
                    cand_encoded = self.tokenizer(cand_sentence, max_length=self.config['max_length'], padding='max_length', truncation=True, return_tensors='pt').to(self.device)
                    cand_outputs = self.model.bert(input_ids=cand_encoded['input_ids'], attention_mask=cand_encoded['attention_mask'])
                    
                    cand_prefix_tokens = self.tokenizer.tokenize(current_sentence[:start_char])
                    cand_word_tokens = self.tokenizer.tokenize(cand_word)
                    cand_start_idx = len(cand_prefix_tokens) + 1
                    cand_end_idx = cand_start_idx + len(cand_word_tokens)
                    
                    cand_seq_len = cand_outputs.last_hidden_state.size(1)
                    cand_start_idx_c = min(cand_start_idx, cand_seq_len - 1)
                    cand_end_idx_c = min(max(cand_end_idx, cand_start_idx_c + 1), cand_seq_len)
                    cand_contextual_embed = cand_outputs.last_hidden_state[0, cand_start_idx_c:cand_end_idx_c].mean(dim=0).cpu()
                    
                    cosine_sim = F.cosine_similarity(orig_contextual_embed.unsqueeze(0), cand_contextual_embed.unsqueeze(0)).item()
                    
                    # Model ranker score
                    feat_context = context_embed.unsqueeze(0).to(self.device)
                    feat_mlm = torch.tensor([[c['mlm_prob']]], dtype=torch.float32).to(self.device)
                    feat_sim = torch.tensor([[cosine_sim]], dtype=torch.float32).to(self.device)
                    feat_delta = torch.tensor([[c['simplicity_delta']]], dtype=torch.float32).to(self.device)
                    
                    final_score = self.model(feat_context, feat_mlm, feat_sim, feat_delta).item()
                    
                scored_candidates.append({
                    'word': cand_word,
                    'mlm_prob': c['mlm_prob'],
                    'cosine': cosine_sim,
                    'delta': c['simplicity_delta'],
                    'score': final_score
                })
                
            scored_candidates.sort(key=lambda x: x['score'], reverse=True)
            for sc in scored_candidates[:5]:
                print(f"  {sc['word']:<10} | {sc['mlm_prob']:.4f}   | {sc['cosine']:.4f}      | {sc['delta']:.4f}    | {sc['score']:.4f}")
                
            # Stage 6: Word Replacement
            winner = scored_candidates[0]
            winner_inflected = inflect_candidate(word, pos, winner['word'])
            
            # Capitalization matching
            if word.isupper():
                winner_final = winner_inflected.upper()
            elif word.istitle():
                winner_final = winner_inflected.title()
            else:
                winner_final = winner_inflected
                
            print(f"Stage 6 - Word Replacement:")
            print(f"  Original Word  : {word}")
            print(f"  Substituted    : {winner_final}")
            
            current_sentence = current_sentence[:start_char] + winner_final + current_sentence[end_char:]
            
            # Update offset shift for subsequent words
            offset_shift += len(winner_final) - len(word)
            
        print("="*60)
        print(f"FINAL OUTPUT SENTENCE: {current_sentence}")
        print("="*60 + "\n")
        return current_sentence

# ---------------------------------------------------------
# Main Execution Entry
# ---------------------------------------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained(CONFIG['bert_model'])
    generator = CandidateGenerator(CONFIG)
    
    print("Loading BenchLS Dataset...")
    full_dataset = LexicalSimplificationDataset(CONFIG, tokenizer, "./")
    print(f"Dataset loaded with {len(full_dataset)} total samples.")
    
    # Group by unique sentence/target to prevent data leakage
    grouped_samples = {}
    for s in full_dataset.samples:
        key = (s['sentence'], s['target_word'], s['start_char'], s['end_char'])
        if key not in grouped_samples:
            grouped_samples[key] = []
        grouped_samples[key].append(s)
        
    keys = list(grouped_samples.keys())
    random.seed(CONFIG['seed'])
    random.shuffle(keys)
    
    num_val = int(len(keys) * CONFIG['val_split'])
    val_keys = keys[:num_val]
    train_keys = keys[num_val:]
    
    # Subset train keys to 150 sentences for fast training on CPU
    train_keys = train_keys[:150]
    
    train_samples = []
    for k in train_keys:
        train_samples.extend(grouped_samples[k])
        
    val_samples = []
    for k in val_keys:
        val_samples.extend(grouped_samples[k])
        
    print(f"Train subset sentences: {len(train_keys)} ({len(train_samples)} samples)")
    print(f"Val sentences: {len(val_keys)} ({len(val_samples)} samples)")
    
    # Initialize Model
    model = LexicalSimplificationModel(CONFIG, tokenizer.vocab_size).to(device)
    
    # Precompute all features
    print("\nPrecomputing Training Features...")
    train_cached = precompute_all_features(train_samples, model, tokenizer, device)
    print("Precomputing Validation Features...")
    val_cached = precompute_all_features(val_samples, model, tokenizer, device)
    
    train_dataset = PrecomputedDataset(train_cached)
    val_dataset = PrecomputedDataset(val_cached)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr_ranker'], weight_decay=CONFIG['weight_decay'])
    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    MODEL_PATH = './best_model.pt'
    
    print("\n=== Starting Ranker Training ===")
    for epoch in range(CONFIG['epochs']):
        epoch_train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, CONFIG['grad_clip'])
        epoch_val_loss = validate(model, val_loader, device)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: Train Loss = {epoch_train_loss:.4f} | Val Loss = {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[Checkpoint Saved] -> {MODEL_PATH}")
            
    print("\n[Training Complete]")
    
    # Save loss curve plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='darkblue', marker='o')
    plt.plot(val_losses, label='Val Loss', color='crimson', marker='x')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.savefig('loss_curves.png')
    print("Saved loss curves plot to 'loss_curves.png'")
    
    # Evaluate model
    print("\n=== Evaluating Validation Performance ===")
    evaluate_model_static(model, val_samples, tokenizer, device, DALE_CHALL_WORDS, OXFORD_WORDS)
    
    # Quick Test
    preprocessor = Preprocessor()
    cwi = ComplexWordIdentifier()
    simplifier = LexicalSimplifier(CONFIG, MODEL_PATH, preprocessor, cwi, generator, tokenizer, device)
    
    print("=== Inference Test ===")
    s1 = "The nature looks elegant today"
    simplifier.simplify(s1)
