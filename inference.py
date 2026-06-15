import os
import sys
import math
import re
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from transformers import BertTokenizer

# Import local modules
from config import CONFIG
from preprocessing import Preprocessor
from contextual_cwi import ComplexWordIdentifier
from word_sense_disambiguation import WordSenseDisambiguator
from candidate_generator import CandidateGenerator
from model import LexicalSimplificationModel

def _should_double_consonant(word: str) -> bool:
    if len(word) < 3:
        return False
    vowels = "aeiou"
    if word[-1] not in vowels and word[-1] not in "wxy":
        if word[-2] in vowels:
            if word[-3] not in vowels:
                return True
    return False

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
        elif target_word.endswith('ss') and cand_lower.endswith('s') and not cand_lower.endswith('ss'):
            return cand_lower + 's'
            
    # 2. Verb Conjugations
    elif target_pos == 'VERB':
        if target_word.endswith('ing'):
            # Present participle
            if not cand_lower.endswith('ing'):
                if cand_lower.endswith('e') and not cand_lower.endswith('ee'):
                    return cand_lower[:-1] + 'ing'
                elif _should_double_consonant(cand_lower):
                    return cand_lower + cand_lower[-1] + 'ing'
                else:
                    return cand_lower + 'ing'
        elif target_word.endswith('ed'):
            # Past tense
            if not cand_lower.endswith('ed'):
                if cand_lower.endswith('e'):
                    return cand_lower + 'd'
                elif cand_lower.endswith('y') and len(cand_lower) > 1 and cand_lower[-2] not in 'aeiou':
                    return cand_lower[:-1] + 'ied'
                elif _should_double_consonant(cand_lower):
                    return cand_lower + cand_lower[-1] + 'ed'
                else:
                    return cand_lower + 'ed'
        elif target_word.endswith('s') and not target_word.endswith('ss'):
            # Third person singular
            if not cand_lower.endswith('s'):
                if cand_lower.endswith(('ch', 'sh', 'x', 'z', 'o')):
                    return cand_lower + 'es'
                elif cand_lower.endswith('y') and len(cand_lower) > 1 and cand_lower[-2] not in 'aeiou':
                    return cand_lower[:-1] + 'ies'
                else:
                    return cand_lower + 's'
                    
    return cand_lower


class LexicalSimplifier:
    """
    LexicalSimplifier implements the end-to-end 6-stage lexical simplification pipeline.
    """
    def __init__(self, config: Dict[str, Any], model_path: str, device: torch.device) -> None:
        """
        Loads all required modules (spaCy, BERT tokenizer, CWI, WSD, Generator, Model weights).
        """
        self.config = config
        self.device = device
        
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_model'])
        
        # Load core trained ranker model
        self.model = LexicalSimplificationModel(config, self.tokenizer.vocab_size).to(device)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded trained ranker weights from {model_path}")
            except Exception as exc:
                print(f"Error loading trained ranker weights: {exc}. Using base/random weights.")
        else:
            print(f"Warning: Checkpoint path '{model_path}' not found. Using random weights.")
            
        self.model.eval()
        
        # Initialize supporting pipeline stages
        self.preprocessor = Preprocessor()
        self.cwi = ComplexWordIdentifier(config, self.tokenizer, self.model, device)
        self.wsd = WordSenseDisambiguator(config, self.tokenizer, self.model.bert, device)
        self.generator = CandidateGenerator(config, self.tokenizer, self.model, device)

    def simplify(self, sentence: str) -> str:
        """
        Runs the 6-stage lexical simplification pipeline sequentially on the input sentence.
        """
        if not sentence.strip():
            return sentence
            
        print("="*60)
        print(f"INPUT SENTENCE : {sentence}")
        print("="*60)
        
        # Stage 1: Preprocessing
        tokens_info = self.preprocessor.preprocess(sentence)
        print("\nStage 1 - Preprocessing:")
        print(f"  Tokens: [{', '.join([f'{t['text']}({t['pos']})' for t in tokens_info])}]")
        
        # Stage 2: Complex Word Identification (CWI)
        complex_words: List[Dict[str, Any]] = []
        print("\nStage 2 - Context-Aware CWI Candidates:")
        
        for tok in tokens_info:
            if tok['is_skippable']:
                continue
                
            is_comp, score = self.cwi.is_complex(sentence, tok['start'], tok['end'], tok['text'], tok['lemma'])
            
            # Print diagnostic features
            print(f"  Word: '{tok['text']}' | Score: {score:.3f} | Complex: {is_comp}")
            
            if is_comp:
                tok['score'] = score
                complex_words.append(tok)
                
        # Sort complex words by their start position to ensure sequential replacement does not corrupt offsets
        complex_words.sort(key=lambda x: x['start'])
        
        if not complex_words:
            print("\nNo complex words identified. Returning original sentence.")
            print("="*60)
            return sentence
            
        print(f"\nProceeding with sequential substitution of {len(complex_words)} complex word(s)...")
        
        current_sentence = sentence
        offset_shift = 0
        
        for index, target in enumerate(complex_words):
            word = target['text']
            pos = target['pos']
            start_char = target['start'] + offset_shift
            end_char = target['end'] + offset_shift
            
            # Offsets boundary sanity checks
            if start_char < 0 or end_char > len(current_sentence):
                continue
                
            # Offset shift adjustment validation
            if current_sentence[start_char:end_char].lower() != word.lower():
                # Locate target word dynamically in modified sentence
                pattern = r'\b' + re.escape(word) + r'\b'
                matches = list(re.finditer(pattern, current_sentence, re.IGNORECASE))
                if matches:
                    # Choose match closest to original start index
                    match = min(matches, key=lambda m: abs(m.start() - start_char))
                    start_char = match.start()
                    end_char = match.end()
                else:
                    # Fallback to character indexing
                    continue
                    
            print(f"\n--- Stage 3, 4, 5, 6 for Word {index+1}/{len(complex_words)}: '{word}' ({pos}) ---")
            
            # Stage 3: Word Sense Disambiguation
            chosen_sense = self.wsd.disambiguate(current_sentence, start_char, end_char, word, pos)
            print(f"Stage 3 - Word Sense Disambiguation:")
            if chosen_sense:
                print(f"  Chosen Sense  : {chosen_sense.name()}")
                print(f"  Definition    : {chosen_sense.definition()}")
            else:
                print("  No synset disambiguated (WordNet missing). Skipping sense alignment.")
                
            # Stage 4: Candidate Generation
            candidates = self.generator.generate(current_sentence, start_char, end_char, word, pos, chosen_sense, self.cwi)
            print(f"Stage 4 - Candidate Generation:")
            if not candidates:
                print("  No simpler candidates found. Skipping replacement.")
                continue
            print(f"  Filtered Candidates (Strictly Simpler): {[c['word'] for c in candidates[:15]]} ... ({len(candidates)} total)")
            
            # Stage 5: Contextual Neural Ranking
            print("Stage 5 - Contextual Neural Ranking:")
            print("  Candidate  | MLM Prob | Context Cos | Sim Delta | FINAL SCORE")
            print("  " + "-"*56)
            
            scored_candidates: List[Tuple[str, float]] = []
            
            for cand in candidates:
                cand_word = cand['word']
                
                # Context embedding for CLS
                encoded_sent = self.tokenizer(
                    current_sentence, 
                    max_length=self.config['max_length'], 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.bert(input_ids=encoded_sent['input_ids'], attention_mask=encoded_sent['attention_mask'])
                    context_embed = outputs.last_hidden_state[0, 0].cpu() # CLS
                    
                # Original word contextual embedding
                prefix_tokens = self.tokenizer.tokenize(current_sentence[:start_char])
                word_tokens = self.tokenizer.tokenize(word)
                orig_start_idx = len(prefix_tokens) + 1
                orig_end_idx = orig_start_idx + len(word_tokens)
                orig_seq_len = outputs.last_hidden_state.size(1)
                
                orig_start_idx_c = min(orig_start_idx, orig_seq_len - 1)
                orig_end_idx_c = min(max(orig_end_idx, orig_start_idx_c + 1), orig_seq_len)
                orig_embed = outputs.last_hidden_state[0, orig_start_idx_c:orig_end_idx_c].mean(dim=0)
                
                # Candidate sentence contextual embedding
                cand_sentence = current_sentence[:start_char] + cand_word + current_sentence[end_char:]
                cand_encoded = self.tokenizer(
                    cand_sentence, 
                    max_length=self.config['max_length'], 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    cand_outputs = self.model.bert(input_ids=cand_encoded['input_ids'], attention_mask=cand_encoded['attention_mask'])
                    
                cand_prefix_tokens = self.tokenizer.tokenize(current_sentence[:start_char])
                cand_word_tokens = self.tokenizer.tokenize(cand_word)
                cand_start_idx = len(cand_prefix_tokens) + 1
                cand_end_idx = cand_start_idx + len(cand_word_tokens)
                
                cand_seq_len = cand_outputs.last_hidden_state.size(1)
                cand_start_idx_c = min(cand_start_idx, cand_seq_len - 1)
                cand_end_idx_c = min(max(cand_end_idx, cand_start_idx_c + 1), cand_seq_len)
                cand_embed = cand_outputs.last_hidden_state[0, cand_start_idx_c:cand_end_idx_c].mean(dim=0)
                
                cosine_sim = F.cosine_similarity(orig_embed.unsqueeze(0), cand_embed.unsqueeze(0)).item()
                
                # Get candidate MLM prob
                masked_sentence = current_sentence[:start_char] + "[MASK]" + current_sentence[end_char:]
                masked_inputs = self.tokenizer(
                    masked_sentence, 
                    max_length=self.config['max_length'], 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt'
                ).to(self.device)
                
                mask_indices = (masked_inputs['input_ids'][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_indices) > 0:
                    mask_idx = mask_indices[0].item()
                    with torch.no_grad():
                        m_outputs = self.model.bert(input_ids=masked_inputs['input_ids'], attention_mask=masked_inputs['attention_mask'])
                        m_feat = m_outputs.last_hidden_state[0, mask_idx]
                        mlm_logits = self.model.mlm_head(m_feat)
                        mlm_probs = F.softmax(mlm_logits, dim=-1)
                else:
                    mlm_probs = torch.zeros(self.tokenizer.vocab_size).to(self.device)
                    
                cand_toks = self.tokenizer.tokenize(cand_word)
                cand_tok_id = self.tokenizer.convert_tokens_to_ids(cand_toks[0]) if cand_toks else self.tokenizer.unk_token_id
                mlm_prob_val = mlm_probs[cand_tok_id].item()
                
                # Model evaluation
                with torch.no_grad():
                    context_t = context_embed.unsqueeze(0).to(self.device)
                    mlm_prob_t = torch.tensor([[mlm_prob_val]], dtype=torch.float32).to(self.device)
                    cosine_sim_t = torch.tensor([[cosine_sim]], dtype=torch.float32).to(self.device)
                    sim_delta_t = torch.tensor([[cand['simplicity_delta']]], dtype=torch.float32).to(self.device)
                    
                    final_score = self.model(context_t, mlm_prob_t, cosine_sim_t, sim_delta_t).item()
                    
                scored_candidates.append((cand_word, final_score))
                
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            for item in scored_candidates[:5]:
                print(f"  {item[0]:10} | {mlm_prob_val:.4f}   | {cosine_sim:.4f}      | {cand['simplicity_delta']:.4f}    | {item[1]:.4f}")
                
            best_candidate = scored_candidates[0][0]
            
            # Stage 6: Word Replacement
            # Inflect candidate to match casing and grammar
            inflected = inflect_candidate(word, pos, best_candidate)
            
            # Preserve capitalization style
            if word.isupper():
                inflected = inflected.upper()
            elif word.istitle() or (len(word) > 0 and word[0].isupper()):
                inflected = inflected.capitalize()
                
            print(f"Stage 6 - Word Replacement:")
            print(f"  Original Word  : {word}")
            print(f"  Substituted    : {inflected}")
            
            # Replace target in sentence
            current_sentence = current_sentence[:start_char] + inflected + current_sentence[end_char:]
            offset_shift += len(inflected) - len(word)
            
        print("="*60)
        print(f"FINAL OUTPUT SENTENCE: {current_sentence}")
        print("="*60)
        return current_sentence


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = CONFIG['best_model_path'] if os.path.exists(CONFIG['best_model_path']) else "./best_model.pt"
    
    print("Loading models and resources (this will take a few seconds)...")
    simplifier = LexicalSimplifier(CONFIG, model_path, device)
    
    print("\nModel resources loaded successfully!\n")
    print("="*60)
    print("                 LEXICAL SIMPLIFICATION RUNNER")
    print("="*60)
    
    while True:
        try:
            test_sentence = input("Enter sentence (or type '1' to exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if test_sentence == '1':
            print("Exiting...")
            break
        if not test_sentence:
            continue
            
        try:
            simplifier.simplify(test_sentence)
        except Exception as e:
            print(f"\nError running inference: {e}")

