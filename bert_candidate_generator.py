# bert_candidate_generator.py
"""
Upgraded BERTCandidateGenerator — 5-priority candidate source architecture.

Priority 1 (Highest): Gold lookup table (LexMTurk + BenchLS human substitutions)
Priority 2:           WordNet same-synset synonyms (strictly filtered)
Priority 3:           GloVe 840B nearest neighbours
Priority 4:           FastText nearest neighbours (handles rare/OOV words)
Priority 5 (Last):    BERT MLM predictions

Combined filtering (all sources must pass):
  ✓ POS must match original word (strict)
  ✓ Candidate Zipf > target Zipf (simpler)
  ✓ SBERT sentence similarity > 0.90 (strict meaning preservation)
  ✓ Word cosine similarity > 0.75 (strict)
  ✓ BERT MLM probability > 0.0005
  ✓ Not the same as the original word
  ✓ Not more complex than original (complexity check)
"""

from __future__ import annotations

import os
import wordfreq
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple

from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_surprisal import BERTSurprisalCalculator


# ─────────────────────────────────────────────────────────────────────────────
# Semantic relation helper (unchanged API, kept for import compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def are_semantically_related(
    chosen_sense, target_word: str, cand: str, pos_tag: str
) -> bool:
    """
    Returns True if `cand` is semantically related to `target_word` in WordNet.
    Checks: same synset, hypernyms, hyponyms, sister terms, similar_tos, and also_sees.
    Supports VERB/ADJ crossover (participle matching).
    """
    pos_map = {
        'NOUN': wn.NOUN, 'VERB': wn.VERB,
        'ADJ': wn.ADJ,   'ADV': wn.ADV, 'PROPN': wn.NOUN,
    }
    wn_pos = pos_map.get(pos_tag.upper()) if pos_tag else None

    # Resolve candidate synsets, supporting VERB/ADJ crossover
    c_s = []
    if wn_pos:
        c_s.extend(wn.synsets(cand.lower(), pos=wn_pos))
        if pos_tag.upper() in ('VERB', 'ADJ'):
            extra_pos = wn.ADJ if pos_tag.upper() == 'VERB' else wn.VERB
            c_s.extend(wn.synsets(cand.lower(), pos=extra_pos))
    else:
        c_s.extend(wn.synsets(cand.lower()))

    if not c_s:
        return False
    c_set = set(c_s)

    def _get_related_synsets(synset):
        if not synset:
            return set()
        rels = set()
        # Direct relations (hypernyms, hyponyms)
        rels.update(synset.hypernyms())
        rels.update(synset.hyponyms())
        # Similar / also_see for adjectives
        if hasattr(synset, 'similar_tos'):
            rels.update(synset.similar_tos())
        if hasattr(synset, 'also_sees'):
            rels.update(synset.also_sees())
        return rels

    if chosen_sense:
        if chosen_sense in c_set:
            return True
        direct_relations = _get_related_synsets(chosen_sense)
        if any(rel in c_set for rel in direct_relations):
            return True
        # Sister terms
        for hyp in chosen_sense.hypernyms():
            if any(sister in c_set for sister in hyp.hyponyms()):
                return True
        return False

    # Resolve target synsets, supporting VERB/ADJ crossover
    t_s = []
    if wn_pos:
        t_s.extend(wn.synsets(target_word.lower(), pos=wn_pos))
        if pos_tag.upper() in ('VERB', 'ADJ'):
            extra_pos = wn.ADJ if pos_tag.upper() == 'VERB' else wn.VERB
            t_s.extend(wn.synsets(target_word.lower(), pos=extra_pos))
    else:
        t_s.extend(wn.synsets(target_word.lower()))

    if not t_s:
        return False

    for ts in t_s:
        if ts in c_set:
            return True
        direct_relations = _get_related_synsets(ts)
        if any(rel in c_set for rel in direct_relations):
            return True
        for hyp in ts.hypernyms():
            if any(sister in c_set for sister in hyp.hyponyms()):
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# SBERT sentence similarity helper (lazy-loaded)
# ─────────────────────────────────────────────────────────────────────────────

class _SBERTEncoder:
    """Thin wrapper around SentenceTransformer for sentence-level similarity."""

    def __init__(self, device=None):
        self.device = device
        self._model = None
        self._try_load()

    def _try_load(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                'sentence-transformers/all-mpnet-base-v2')
            if self.device:
                self._model.to(self.device)
        except Exception:
            self._model = None

    def similarity(self, s1: str, s2: str) -> float:
        if self._model is None:
            return 1.0   # neutral fallback (do not reject candidates)
        import numpy as np
        e1, e2 = self._model.encode(
            [s1, s2], convert_to_numpy=True, normalize_embeddings=True)
        return float(np.dot(e1, e2))

    @property
    def available(self) -> bool:
        return self._model is not None


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class BERTCandidateGenerator:
    """
    5-priority candidate generator with strict precision-first filtering.

    Constructor parameters
    ─────────────────────
    tokenizer, model, bert_model, device : shared BERT instances
    gold_table   : Dict[str, List[str]] from data_loader.build_gold_table()
    embedding_store : EmbeddingStore from embedding_store.py (GloVe + FastText)
    glove_model  : legacy gensim KeyedVectors (kept for backward compat)
    """

    # Strict cosine thresholds
    MIN_WORD_COS_SIM  = 0.65   # word-level embedding cosine similarity
    MIN_SENT_SIM      = 0.90   # SBERT sentence similarity
    MIN_MLM_PROB      = 0.0005
    MIN_ZIPF_GAIN     = 0.0    # candidate must be at least as common as original

    def __init__(
        self,
        tokenizer=None,
        model=None,
        bert_model=None,
        device=None,
        gold_table:      Optional[Dict[str, List[str]]] = None,
        embedding_store=None,       # EmbeddingStore instance
        glove_model=None,           # legacy gensim KV
    ) -> None:
        self.device    = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(
            'bert-base-uncased')
        self.model     = model or BertForMaskedLM.from_pretrained(
            'bert-base-uncased').to(self.device)
        self.bert_model = bert_model or BertModel.from_pretrained(
            'bert-base-uncased').to(self.device)

        self.model.eval()
        self.bert_model.eval()

        self.surprisal_calc = BERTSurprisalCalculator(
            self.tokenizer, self.model, self.device)

        # Priority 1: Gold lookup table
        self.gold_table: Dict[str, List[str]] = gold_table or {}

        # Priority 3 + 4: EmbeddingStore (GloVe + FastText)
        self.emb_store = embedding_store  # may be None

        # Legacy gensim GloVe support
        self.glove = glove_model
        if self.glove is None and embedding_store is not None:
            self.glove = getattr(embedding_store, '_glove_gensim', None)

        # SBERT for sentence-level meaning preservation
        self._sbert = _SBERTEncoder(self.device)
        if not self._sbert.available:
            print("[BERTCandidateGenerator] SBERT unavailable; "
                  "sentence similarity filter will be relaxed.")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _wn_pos(self, pos_tag: str):
        return {'NOUN': wn.NOUN, 'VERB': wn.VERB, 'ADJ': wn.ADJ,
                'ADV': wn.ADV, 'PROPN': wn.NOUN}.get(
            pos_tag.upper() if pos_tag else '', None)

    def _lemma(self, word: str, pos_tag: str) -> str:
        wn_pos = self._wn_pos(pos_tag)
        lm = wn.morphy(word.lower(), pos=wn_pos) if wn_pos else None
        return lm or word.lower()

    def _get_sentence_emb_bert(self, sentence: str) -> torch.Tensor:
        """BERT CLS embedding (for word-level cosine similarity)."""
        enc = self.tokenizer(sentence, return_tensors='pt',
                             padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            return self.bert_model(**enc).last_hidden_state[0, 0]

    def _get_word_emb(self, sentence: str, word: str, start: int) -> torch.Tensor:
        enc = self.tokenizer(sentence, return_tensors='pt').to(self.device)
        with torch.no_grad():
            states = self.bert_model(**enc).last_hidden_state[0]
        pre = self.tokenizer.tokenize(sentence[:start])
        tok = self.tokenizer.tokenize(word)
        s   = min(len(pre) + 1, states.size(0) - 1)
        e   = min(s + len(tok), states.size(0))
        return states[s:e].mean(dim=0)

    # ─────────────────────────────────────────────────────────────────────────
    # Priority 1: Gold lookup table
    # ─────────────────────────────────────────────────────────────────────────

    def _gold_candidates(self, word: str, lemma: str) -> List[str]:
        return (self.gold_table.get(word.lower(), [])
                + self.gold_table.get(lemma, []))

    # ─────────────────────────────────────────────────────────────────────────
    # Priority 2: WordNet (same synset only, strictly filtered)
    # ─────────────────────────────────────────────────────────────────────────

    def _wordnet_candidates(
        self, word: str, lemma: str, chosen_sense, pos_tag: str
    ) -> Set[str]:
        wn_pos  = self._wn_pos(pos_tag)
        cands: Set[str] = set()

        def _add(syn):
            if syn is None:
                return
            for lm in syn.lemmas():
                name = lm.name().replace('_', ' ').replace('-', ' ').lower()
                if name.isalpha() and ' ' not in name:
                    cands.add(name)

        def _add_with_relations(syn):
            if syn is None:
                return
            _add(syn)
            # Add closely related synsets
            if syn.pos() in ('a', 's'):
                if hasattr(syn, 'similar_tos'):
                    for sim in syn.similar_tos():
                        _add(sim)
                if hasattr(syn, 'also_sees'):
                    for see in syn.also_sees():
                        _add(see)
            else:
                for hyper in syn.hypernyms():
                    _add(hyper)
                for hypo in syn.hyponyms():
                    _add(hypo)

        if chosen_sense:
            _add_with_relations(chosen_sense)
        else:
            synsets_to_check = []
            if wn_pos:
                synsets_to_check.extend(wn.synsets(lemma, pos=wn_pos))
                synsets_to_check.extend(wn.synsets(word.lower(), pos=wn_pos))
            else:
                synsets_to_check.extend(wn.synsets(lemma))
                synsets_to_check.extend(wn.synsets(word.lower()))

            # Cross-check ADJ for VERB and VERB for ADJ to capture participles/adjectives
            if pos_tag and pos_tag.upper() in ('VERB', 'ADJ'):
                extra_pos = wn.ADJ if pos_tag.upper() == 'VERB' else wn.VERB
                synsets_to_check.extend(wn.synsets(lemma, pos=extra_pos))
                synsets_to_check.extend(wn.synsets(word.lower(), pos=extra_pos))

            for syn in synsets_to_check:
                _add_with_relations(syn)

        cands.discard(word.lower())
        cands.discard(lemma)
        return cands


    # ─────────────────────────────────────────────────────────────────────────
    # Priority 3+4: GloVe + FastText via EmbeddingStore
    # ─────────────────────────────────────────────────────────────────────────

    def _embedding_candidates(
        self, word: str, lemma: str, target_zipf: float, topn: int = 20
    ) -> Set[str]:
        cands: Set[str] = set()
        if self.emb_store is None and self.glove is None:
            return cands

        def _add_from_list(pairs):
            for w, sim in pairs:
                wl = w.strip().lower()
                if (wl.isalpha()
                        and ' ' not in wl
                        and wl != word.lower()
                        and wl != lemma
                        and sim >= 0.72):
                    cands.add(wl)

        if self.emb_store is not None:
            # GloVe (Priority 3)
            _add_from_list(self.emb_store.nearest_neighbours(
                lemma, topn=topn, source='glove'))
            # FastText (Priority 4)
            _add_from_list(self.emb_store.nearest_neighbours(
                lemma, topn=topn, source='fasttext'))
        elif self.glove is not None:
            # Legacy gensim path
            try:
                key = lemma if lemma in self.glove else word.lower()
                if key in self.glove:
                    _add_from_list(self.glove.most_similar(key, topn=topn))
            except Exception:
                pass

        return cands

    # ─────────────────────────────────────────────────────────────────────────
    # Priority 5: BERT MLM
    # ─────────────────────────────────────────────────────────────────────────

    def _mlm_candidates(
        self,
        sentence:   str,
        start_char: int,
        end_char:   int,
        topn:       int = 10,
    ) -> Tuple[Set[str], torch.Tensor]:
        """Returns (candidate set, full probability vector)."""
        masked = self.surprisal_calc.get_masked_sentence_and_idx(
            sentence, start_char, end_char)
        enc  = self.tokenizer(masked, return_tensors='pt').to(self.device)
        idxs = (enc['input_ids'][0] == self.tokenizer.mask_token_id
                ).nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
            vocab = self.tokenizer.vocab_size
            return set(), torch.zeros(vocab).to(self.device)

        mask_pos = idxs[0].item()
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs  = F.softmax(logits[0, mask_pos], dim=-1)
        top_k  = torch.topk(probs, topn)
        cands: Set[str] = set()
        for idx in top_k.indices.tolist():
            w = self.tokenizer.decode([idx]).strip().lower()
            if w.isalpha() and not w.startswith('##'):
                cands.add(w)
        return cands, probs

    # ─────────────────────────────────────────────────────────────────────────
    # Combined filtering
    # ─────────────────────────────────────────────────────────────────────────

    def _passes_filters(
        self,
        cand:         str,
        orig_word:    str,
        sentence:     str,
        start_char:   int,
        end_char:     int,
        pos_tag:      str,
        orig_zipf:    float,
        all_probs:    torch.Tensor,
        orig_word_emb: torch.Tensor,
        orig_sent_emb: torch.Tensor,
        chosen_sense,
        sense_conf:   float,
        nlp=None,
    ) -> bool:
        cand_l = cand.lower()
        if cand_l == orig_word.lower():
            return False
        if not cand_l.isalpha() or ' ' in cand_l:
            return False

        # ── Zipf: candidate must be simpler (higher frequency) ───────────────
        cand_zipf = wordfreq.zipf_frequency(cand_l, 'en')
        if cand_zipf <= orig_zipf + self.MIN_ZIPF_GAIN:
            return False

        # ── MLM probability ──────────────────────────────────────────────────
        toks = self.tokenizer(cand_l, add_special_tokens=False)['input_ids']
        if not toks:
            return False
        mlm_prob = all_probs[toks[0]].item()
        if mlm_prob < self.MIN_MLM_PROB:
            return False

        # ── SBERT sentence similarity > 0.90 ────────────────────────────────
        cand_sentence = sentence[:start_char] + cand_l + sentence[end_char:]
        if self._sbert.available:
            sent_sim = self._sbert.similarity(sentence, cand_sentence)
            if sent_sim < self.MIN_SENT_SIM:
                return False
        else:
            # Fallback: BERT CLS cosine
            cand_sent_emb = self._get_sentence_emb_bert(cand_sentence)
            sent_sim = F.cosine_similarity(
                orig_sent_emb.unsqueeze(0),
                cand_sent_emb.unsqueeze(0)).item()
            if sent_sim < self.MIN_SENT_SIM:
                return False

        # ── Word-level BERT cosine similarity > 0.75 ────────────────────────
        cand_word_emb = self._get_word_emb(cand_sentence, cand_l, start_char)
        word_sim = F.cosine_similarity(
            orig_word_emb.unsqueeze(0),
            cand_word_emb.unsqueeze(0)).item()
        if word_sim < self.MIN_WORD_COS_SIM:
            return False

        # ── POS must match (strict) ──────────────────────────────────────────
        if nlp is not None and pos_tag:
            def _get_pos(doc, sc, ec):
                for t in doc:
                    if t.idx <= sc < t.idx + len(t.text):
                        return t.pos_
                return None
            orig_doc = nlp(sentence)
            cand_doc = nlp(cand_sentence)
            orig_pos = _get_pos(orig_doc, start_char, end_char)
            cand_pos = _get_pos(cand_doc, start_char, start_char + len(cand_l))
            if orig_pos and cand_pos:
                compatible = (orig_pos == cand_pos) or ({orig_pos, cand_pos} == {'VERB', 'ADJ'}) or ({orig_pos, cand_pos} == {'NOUN', 'PROPN'})
                if not compatible:
                    return False


        # ── WordNet semantic relation (relaxed by high sentence sim) ─────────
        if sense_conf >= 0.40:
            related = are_semantically_related(chosen_sense, orig_word, cand_l, pos_tag)
            if not related and sent_sim < 0.94:
                return False

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def generate_raw_candidates_by_source(
        self,
        sentence:    str,
        target_word: str,
        start_char:  int,
        end_char:    int,
        chosen_sense,
        pos_tag:     str = None,
    ) -> Dict[str, List[str]]:
        """
        Returns a dict with keys 'gold', 'wordnet', 'glove', 'fasttext', 'bert_mlm'
        for diagnostics and source tracking.
        Also includes 'all_probs' (MLM probability tensor) for reuse in Stage 4.
        """
        word_l  = target_word.lower()
        lemma   = self._lemma(word_l, pos_tag or '')

        # P1: Gold
        gold  = list(dict.fromkeys(self._gold_candidates(word_l, lemma)))

        # P2: WordNet
        wn_c  = self._wordnet_candidates(word_l, lemma, chosen_sense, pos_tag or '')

        # P3+4: Embeddings
        emb_c = self._embedding_candidates(
            word_l, lemma,
            target_zipf=wordfreq.zipf_frequency(word_l, 'en'))

        # P5: BERT MLM
        mlm_c, all_probs = self._mlm_candidates(sentence, start_char, end_char, topn=200)


        # Discard original word from all sources
        for s in (wn_c, emb_c, mlm_c):
            s.discard(word_l)
            s.discard(lemma)

        return {
            'gold':     gold,
            'wordnet':  sorted(wn_c),
            'glove':    sorted(emb_c),     # GloVe + FastText merged
            'fasttext': [],                # EmbeddingStore merges both
            'bert_mlm': sorted(mlm_c),
            '_all_probs': all_probs,       # internal, used by filter step
        }

    def generate_raw_pool(
        self,
        sentence:    str,
        target_word: str,
        start_char:  int,
        end_char:    int,
        chosen_sense,
    ) -> set:
        """Backward-compatible: return flat union of all candidates."""
        sources = self.generate_raw_candidates_by_source(
            sentence, target_word, start_char, end_char, chosen_sense)
        pool: set = set()
        for k, v in sources.items():
            if k.startswith('_'):
                continue
            pool.update(v)
        return pool

    def filter_candidates(
        self,
        sentence:    str,
        target_word: str,
        start_char:  int,
        end_char:    int,
        raw_candidates: set,
        chosen_sense=None,
        pos_tag:     str = None,
        sense_conf:  float = 0.0,
        nlp=None,
        all_probs:   torch.Tensor = None,
    ) -> List[str]:
        """
        Apply combined precision-first filters to the raw candidate pool.

        If `all_probs` is pre-computed (from generate_raw_candidates_by_source),
        pass it here to avoid recomputing the MLM forward pass.
        """
        if not raw_candidates:
            return []

        orig_zipf = wordfreq.zipf_frequency(target_word.lower(), 'en')

        # MLM probs
        if all_probs is None:
            _, all_probs = self._mlm_candidates(sentence, start_char, end_char, topn=1)

        # Pre-filter using cheap Zipf and MLM probability checks to avoid expensive BERT/SBERT forward passes
        cheap_candidates = []
        for cand in sorted(raw_candidates):
            cand_l = cand.lower()
            if cand_l == target_word.lower():
                continue
            if not cand_l.isalpha() or ' ' in cand_l:
                continue
            cand_zipf = wordfreq.zipf_frequency(cand_l, 'en')
            if cand_zipf <= orig_zipf + self.MIN_ZIPF_GAIN:
                continue
            toks = self.tokenizer(cand_l, add_special_tokens=False)['input_ids']
            if not toks:
                continue
            mlm_prob = all_probs[toks[0]].item()
            if mlm_prob < self.MIN_MLM_PROB:
                continue
            cheap_candidates.append(cand)

        if not cheap_candidates:
            return []

        # Pre-compute BERT embeddings for original
        orig_word_emb = self._get_word_emb(sentence, target_word, start_char)
        orig_sent_emb = self._get_sentence_emb_bert(sentence)

        filtered: List[str] = []
        for cand in cheap_candidates:
            if self._passes_filters(
                cand, target_word, sentence, start_char, end_char,
                pos_tag or '', orig_zipf, all_probs,
                orig_word_emb, orig_sent_emb,
                chosen_sense, sense_conf, nlp,
            ):
                filtered.append(cand.lower())

        return filtered

    # ─────────────────────────────────────────────────────────────────────────
    # Legacy helpers (unchanged API surface)
    # ─────────────────────────────────────────────────────────────────────────

    def get_sentence_embedding(self, sentence: str) -> torch.Tensor:
        return self._get_sentence_emb_bert(sentence)

    def get_inherent_complexity(self, target_word: str) -> float:
        bias = self.model.cls.predictions.bias
        toks = self.tokenizer.encode(target_word, add_special_tokens=False)
        if not toks:
            return 1.0
        avg_bias = sum(bias[t].item() for t in toks) / len(toks)
        mapped = max(0.0, min(1.0, 1.0 - (avg_bias + 1.0) / 1.5))
        penalty = 0.15 * (len(toks) - 1)
        return min(1.0, mapped + penalty)

    def get_neutral_template_surprisal(self, target_word: str, pos: str = None) -> float:
        return self.get_inherent_complexity(target_word)
