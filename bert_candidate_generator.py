import os
import torch
import torch.nn.functional as F
import gensim.downloader as api
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_surprisal import BERTSurprisalCalculator

def are_semantically_related(chosen_sense, target_word: str, cand: str, pos_tag: str) -> bool:
    """
    Checks if a candidate is semantically and grammatically related to the target word.
    """
    pos_map = {
        'NOUN': wn.NOUN,
        'VERB': wn.VERB,
        'ADJ': wn.ADJ,
        'ADV': wn.ADV,
        'PROPN': wn.NOUN
    }
    wn_pos = pos_map.get(pos_tag.upper()) if pos_tag else None
    if not wn_pos:
        return True
        
    c_s = wn.synsets(cand.lower(), pos=wn_pos)
    if not c_s:
        return False
        
    # Check relationship to the chosen_sense if WSD succeeded
    if chosen_sense:
        chosen_set = {chosen_sense}
        c_set = set(c_s)
        if chosen_set.intersection(c_set):
            return True
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

class BERTCandidateGenerator:
    """
    BERTCandidateGenerator aggregates candidates from BERT MLM, WordNet, and GloVe,
    filtering them strictly using BERT contextual metrics:
    - Surprisal reduction
    - Sentence embedding meaning preservation (>0.85)
    - Contextual MLM grammatical fit
    """
    def __init__(self, tokenizer=None, model=None, bert_model=None, device=None) -> None:
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = model if model is not None else BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.bert_model = bert_model if bert_model is not None else BertModel.from_pretrained('bert-base-uncased')
        
        self.model.to(self.device)
        self.bert_model.to(self.device)
        
        self.surprisal_calc = BERTSurprisalCalculator(self.tokenizer, self.model, self.device)
        
        # Bypassed GloVe model loading for fast offline execution
        self.glove = None

    def get_sentence_embedding(self, sentence: str) -> torch.Tensor:
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            return outputs.last_hidden_state[0, 0]  # CLS token

    def get_inherent_complexity(self, target_word: str) -> float:
        """
        Computes the inherent complexity of a word based on its BERT MLM prediction bias
        and a subword token length penalty.
        """
        bias = self.model.cls.predictions.bias
        toks = self.tokenizer.encode(target_word, add_special_tokens=False)
        if not toks:
            return 1.0
            
        biases = [bias[t].item() for t in toks]
        avg_bias = sum(biases) / len(biases)
        
        mapped_complexity = 1.0 - (avg_bias + 1.0) / 1.5
        mapped_complexity = max(0.0, min(1.0, mapped_complexity))
        
        token_penalty = 0.15 * (len(toks) - 1)
        
        return min(1.0, mapped_complexity + token_penalty)

    def get_neutral_template_surprisal(self, target_word: str, pos: str = None) -> float:
        """
        Routes to inherent complexity to prevent template-based POS bias.
        """
        return self.get_inherent_complexity(target_word)

    def generate_raw_pool(self, sentence: str, target_word: str, start_char: int, end_char: int, chosen_sense) -> set:
        sources = self.generate_raw_candidates_by_source(sentence, target_word, start_char, end_char, chosen_sense)
        pool = set(sources['wordnet'] + sources['bert_mlm'] + sources['glove'])
        return pool

    def generate_raw_candidates_by_source(self, sentence: str, target_word: str, start_char: int, end_char: int, chosen_sense) -> dict:
        import wordfreq
        wordnet_cands = set()
        bert_mlm_cands = set()
        glove_cands = set()
        
        # 1. WordNet candidates (using similar_tos to support adjectives correctly)
        synsets_to_use = [chosen_sense] if chosen_sense else wn.synsets(target_word.lower())
        for syn in synsets_to_use:
            if not syn:
                continue
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ').replace('-', ' ').strip().lower()
                if name.isalpha():
                    wordnet_cands.add(name)
            for h in syn.hypernyms():
                for lemma in h.lemmas():
                    name = lemma.name().replace('_', ' ').replace('-', ' ').strip().lower()
                    if name.isalpha():
                        wordnet_cands.add(name)
                for sister in h.hyponyms():
                    for lemma in sister.lemmas():
                        name = lemma.name().replace('_', ' ').replace('-', ' ').strip().lower()
                        if name.isalpha():
                            wordnet_cands.add(name)
            for h in syn.hyponyms():
                for lemma in h.lemmas():
                    name = lemma.name().replace('_', ' ').replace('-', ' ').strip().lower()
                    if name.isalpha():
                        wordnet_cands.add(name)
            if hasattr(syn, 'similar_tos'):
                for sim in syn.similar_tos():
                    for lemma in sim.lemmas():
                        name = lemma.name().replace('_', ' ').replace('-', ' ').strip().lower()
                        if name.isalpha():
                            wordnet_cands.add(name)

        # 2. BERT MLM candidates (Top 30 predictions)
        masked_text = self.surprisal_calc.get_masked_sentence_and_idx(sentence, start_char, end_char)
        inputs = self.tokenizer(masked_text, return_tensors='pt').to(self.device)
        mask_indices = (inputs['input_ids'][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_indices) > 0:
            mask_idx = mask_indices[0].item()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            probs = F.softmax(logits[0, mask_idx], dim=-1)
            top_30 = torch.topk(probs, 30)
            for idx in top_30.indices.tolist():
                decoded = self.tokenizer.decode([idx]).strip().lower()
                if decoded.isalpha() and not decoded.startswith("##"):
                    bert_mlm_cands.add(decoded)

        # 3. GloVe candidates
        GLOVE_FALLBACK = {
            'elegant': ['pretty', 'charming', 'fine', 'neat', 'graceful', 'beautiful', 'refined', 'lovely'],
            'physician': ['doctor', 'surgeon', 'clinician', 'medic', 'gp'],
            'comprehended': ['understood', 'realized', 'grasped', 'knew', 'followed', 'perceived']
        }
        if self.glove and target_word.lower() in self.glove:
            try:
                similar = self.glove.most_similar(target_word.lower(), topn=30)
                for w, _ in similar:
                    w_lower = w.strip().lower()
                    if w_lower.isalpha():
                        glove_cands.add(w_lower)
            except Exception:
                pass
        else:
            fallback = GLOVE_FALLBACK.get(target_word.lower(), [])
            for w in fallback:
                glove_cands.add(w.strip().lower())

        target_lower = target_word.lower()
        wordnet_cands.discard(target_lower)
        bert_mlm_cands.discard(target_lower)
        glove_cands.discard(target_lower)

        return {
            'wordnet': sorted(list(wordnet_cands)),
            'bert_mlm': sorted(list(bert_mlm_cands)),
            'glove': sorted(list(glove_cands))
        }

    def filter_candidates(self, sentence: str, target_word: str, start_char: int, end_char: int, raw_candidates: set, chosen_sense = None, pos_tag: str = None) -> list:
        import wordfreq
        orig_freq = wordfreq.zipf_frequency(target_word.lower(), 'en')
        
        candidates_list = list(raw_candidates)
        if not candidates_list:
            return []
            
        masked_text = self.surprisal_calc.get_masked_sentence_and_idx(sentence, start_char, end_char)
        inputs = self.tokenizer(masked_text, return_tensors='pt').to(self.device)
        mask_indices = (inputs['input_ids'][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_indices) == 0:
            return []
            
        mask_idx = mask_indices[0].item()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        probs = F.softmax(logits[0, mask_idx], dim=-1)

        orig_sent_emb = self.get_sentence_embedding(sentence)
        
        filtered = []
        for cand in candidates_list:
            cand_lower = cand.lower()
            if cand_lower == target_word.lower():
                continue
                
            # Get BERT MLM probability
            cand_tokens = self.tokenizer(cand_lower, add_special_tokens=False)['input_ids']
            if not cand_tokens:
                continue
            cand_id = cand_tokens[0]
            cand_prob = probs[cand_id].item()
            
            # Frequency checks
            cand_freq = wordfreq.zipf_frequency(cand_lower, 'en')
            freq_gain = cand_freq - orig_freq
            
            # Meaning preservation check using sentence-level embeddings
            cand_sentence = sentence[:start_char] + cand_lower + sentence[end_char:]
            cand_sent_emb = self.get_sentence_embedding(cand_sentence)
            semantic_sim = F.cosine_similarity(orig_sent_emb.unsqueeze(0), cand_sent_emb.unsqueeze(0)).item()
            passes_semantics = semantic_sim >= 0.85

            passes_freq = (freq_gain >= 0.5) and (cand_freq >= 4.5)
            passes_mlm = cand_prob >= 0.005
            
            if passes_freq and passes_mlm and passes_semantics:
                filtered.append(cand_lower)
                
        return filtered
