# dynamic_cwi.py
"""
Upgraded DynamicContextualCWI — precision-first ensemble CWI module.

Decision pipeline (3 signals + 2 safety constraints):
─────────────────────────────────────────────────────
Signal 1 (55%) — BERT CWI classifier score
    A fine-tuned BERT binary classifier trained on LexMTurk + BenchLS.
    At startup, if no fine-tuned weights are found the module falls back
    to the inherent-bias + surprisal scorer from bert_complexity.py
    and estimates a "pseudo-classifier" score.

Signal 2 (35%) — SBERT semantic drift
    Sentence-BERT (all-mpnet-base-v2) compares the word's embedding in its
    actual context vs. a neutral concrete template "I see the [word]."
    High drift → abstract / figurative usage → complex.
    This specifically fixes "nature of the person" (figurative) vs.
    "nature outside" (literal).

Signal 3 (10%) — Zipf frequency adjustment
    Asymmetric: rare words get a small bonus, common words a stronger
    penalty.  Zipf ≥ 5.5 → hard SIMPLE override (never reaches scoring).

Ensemble score = 0.55 × bert_score + 0.35 × sbert_drift + 0.10 × zipf_adj

Safety constraint A — Figurative override (critical fix):
    IF sbert_drift > DRIFT_OVERRIDE_THRESHOLD (0.45):
        → Always COMPLEX, regardless of ensemble score or Zipf.
    This catches abstract senses even when the classifier is unsure.

Safety constraint B — Hard Zipf ceiling:
    IF zipf_frequency ≥ COMMON_WORD_ZIPF_CEIL (5.5):
        → Always SIMPLE.  Very common words ("go", "make", "good") cannot
          be complex by definition in a precision-first system.

Threshold — Dynamic, per sentence:
    effective_threshold = max(MIN_DYNAMIC_THRESHOLD,
                              mean_ensemble + K_STD × std_ensemble)
    K_STD = 0.6 (lower than 1.2 used in pure BERT mode because the
    ensemble is already better-calibrated, so we can afford a softer
    gate while still prioritising precision).
    Fallback for short sentences (< 3 content words): 0.38.
"""

from __future__ import annotations

import os
import re
import numpy as np
import wordfreq
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional

from transformers import (
    BertTokenizer, BertForMaskedLM, BertModel,
    BertForSequenceClassification,
)
from bert_complexity import BERTComplexityScorer


# ─────────────────────────────────────────────────────────────────────────────
# Figurative Pattern Detector
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Structural Abstraction Detector
# ─────────────────────────────────────────────────────────────────────────────

class StructuralAbstractionDetector:
    """
    SpaCy dependency-parse detector for abstract / figurative constructions.
    Supports:
      - Pattern A: Metaphorical Head Noun (NOUN + prep-of + abstract noun)
      - Pattern B: Abstract Modifier (modifies abstract concept noun)
      - Pattern C: Figurative Predicate (verb with abstract subject/object)
    """

    ABSTRACT_NOUN_LIST = {
        "poverty", "society", "law", "justice", "truth", "nature", "life", "death",
        "success", "failure", "power", "freedom", "knowledge", "wisdom", "faith", "hope",
        "reality", "existence", "humanity", "person", "conflict", "problem", "doubt",
        "responsibility", "democracy", "grief", "crisis", "injustice", "evil",
        "resistance", "darkness", "corruption", "war", "character", "personality", "essence"
    }

    def __init__(self, nlp=None, config=None) -> None:
        self.nlp = nlp
        self.cfg = config

    def detect(self, doc, token) -> Tuple[str, float, float]:
        """
        Detect Pattern A, B, or C for the given token in parsed doc.
        Returns (pattern_name, structural_score, pattern_boost)
        """
        if self.nlp is None or token is None:
            return "None", 0.0, 0.0

        boost_a = getattr(self.cfg, 'structural_boost_pattern_a', 0.15)
        boost_b = getattr(self.cfg, 'structural_boost_pattern_b', 0.10)
        boost_c = getattr(self.cfg, 'structural_boost_pattern_c', 0.12)

        # ── Pattern A: Metaphorical Head Noun
        # Noun governing a prepositional phrase with "of" where obj is abstract
        if token.pos_ == 'NOUN':
            for child in token.children:
                if child.dep_ == 'prep' and child.text.lower() == 'of':
                    for prep_child in child.children:
                        if prep_child.dep_ in ('pobj', 'obj') and prep_child.text.lower() in self.ABSTRACT_NOUN_LIST:
                            return "Pattern_A", 1.0, boost_a

        # ── Pattern B: Abstract Modifier
        # Modifies an abstract concept noun
        if token.dep_ in ('amod', 'compound', 'nmod', 'poss') and token.head.pos_ == 'NOUN':
            if token.head.text.lower() in self.ABSTRACT_NOUN_LIST:
                return "Pattern_B", 0.7, boost_b

        # ── Pattern C: Figurative Predicate
        # Appears in predicate position with abstract subject/object
        if token.pos_ in ('VERB', 'AUX'):
            for child in token.children:
                if child.dep_ in ('nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr') and (
                    child.text.lower() in self.ABSTRACT_NOUN_LIST or
                    any(grandchild.text.lower() in self.ABSTRACT_NOUN_LIST for grandchild in child.children)
                ):
                    return "Pattern_C", 0.5, boost_c

        return "None", 0.0, 0.0



# ─────────────────────────────────────────────────────────────────────────────
# BERT CWI Classifier (fine-tunable)
# ─────────────────────────────────────────────────────────────────────────────

class BERTCWIClassifier:
    """
    Binary BERT sequence classifier for CWI.

    If fine-tuned weights exist at `weights_path`, they are loaded and the
    classifier runs a proper forward pass.

    If no weights exist, it falls back to the pseudo-classifier from
    BERTComplexityScorer (inherent bias + surprisal), ensuring the system
    works out-of-the-box before fine-tuning.

    Fine-tuning:
        trainer = BERTCWITrainer(classifier)
        trainer.train(pairs, epochs=3, batch_size=8)
        trainer.save('cwi_bert.pt')
    """

    WEIGHTS_PATH = 'cwi_bert.pt'

    def __init__(
        self,
        tokenizer=None,
        mlm_model=None,
        bert_model=None,
        device=None,
    ) -> None:
        self.device    = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(
            'bert-base-uncased')
        self.mlm_model = mlm_model
        self.bert_model = bert_model

        # Try loading fine-tuned classifier
        self.classifier: Optional[BertForSequenceClassification] = None
        self._fallback_scorer = BERTComplexityScorer(
            tokenizer, mlm_model, bert_model, device)

        if os.path.exists(self.WEIGHTS_PATH):
            try:
                clf = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased', num_labels=2)
                clf.load_state_dict(
                    torch.load(self.WEIGHTS_PATH, map_location=self.device))
                clf.to(self.device)
                clf.eval()
                self.classifier = clf
                print(f"[BERTCWIClassifier] Loaded fine-tuned weights: "
                      f"{self.WEIGHTS_PATH}")
            except Exception as e:
                print(f"[BERTCWIClassifier] Could not load weights "
                      f"({e}). Using fallback scorer.")
        else:
            print("[BERTCWIClassifier] No fine-tuned weights found. "
                  "Using BERTComplexityScorer fallback. "
                  f"Train and save to '{self.WEIGHTS_PATH}' to upgrade.")

    def score(
        self,
        sentence:   str,
        word:       str,
        start_char: int,
        end_char:   int,
        pos:        str = None,
    ) -> float:
        """
        Returns P(complex) ∈ [0.0, 1.0].

        Fine-tuned path: encodes "[CLS] sentence [SEP]" with the target word
        highlighted by surrounding it with special markers in a second
        segment (token-type = 1).  The classifier logits[:, 1] is sigmoid'd
        to obtain P(complex).

        Fallback path: returns the BERTComplexityScorer composite score
        (inherent + drift + spread) which is already in [0, 1].
        """
        if self.classifier is not None:
            return self._classify(sentence, word, start_char, end_char)
        return self._fallback_scorer.compute_complexity_score(
            sentence, word, start_char, end_char, pos)

    def _classify(
        self, sentence: str, word: str, start_char: int, end_char: int
    ) -> float:
        """Run fine-tuned BertForSequenceClassification."""
        # Mark target word with simple delimiters so the classifier attends to it
        marked = (sentence[:start_char] + '§' + word + '§'
                  + sentence[end_char:])
        enc = self.tokenizer(
            marked,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.classifier(**enc).logits     # (1, 2)
        prob_complex = torch.softmax(logits, dim=-1)[0, 1].item()
        return prob_complex


# ─────────────────────────────────────────────────────────────────────────────
# BERT CWI Fine-tuning Trainer
# ─────────────────────────────────────────────────────────────────────────────

class BERTCWITrainer:
    """
    Fine-tunes BertForSequenceClassification on binary CWI labels.

    Usage:
        from data_loader import load_cwi_training_pairs
        pairs = load_cwi_training_pairs('lex_mturk.txt', 'BenchLS.txt')

        classifier = BERTCWIClassifier(tokenizer, mlm_model, bert_model, device)
        trainer    = BERTCWITrainer(classifier, device)
        trainer.train(pairs, epochs=3, batch_size=8)
        trainer.save('cwi_bert.pt')
    """

    def __init__(self, classifier: BERTCWIClassifier, device=None) -> None:
        self.clf    = classifier
        self.device = device or classifier.device

    def train(
        self,
        pairs:      List[Tuple[str, str, int, int, int]],
        epochs:     int  = 3,
        batch_size: int  = 8,
        lr:         float = 2e-5,
        val_split:  float = 0.1,
    ) -> None:
        """
        pairs = [(sentence, word, start_char, end_char, label), ...]

        Loss: BCEWithLogitsLoss (handles class imbalance well).
        Validation: random 10% split by default.
        """
        import random
        from torch.optim import AdamW

        # Build or reset the classifier model
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=2).to(self.device)
        tok   = self.clf.tokenizer

        random.shuffle(pairs)
        n_val  = max(1, int(len(pairs) * val_split))
        val_p  = pairs[:n_val]
        train_p = pairs[n_val:]

        optimizer = AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print(f"[BERTCWITrainer] Training {len(train_p)} samples, "
              f"validating on {len(val_p)} | epochs={epochs} lr={lr}")

        for epoch in range(1, epochs + 1):
            model.train()
            random.shuffle(train_p)
            total_loss = 0.0
            for i in range(0, len(train_p), batch_size):
                batch = train_p[i: i + batch_size]
                texts  = []
                labels = []
                for sentence, word, sc, ec, label in batch:
                    marked = sentence[:sc] + '§' + word + '§' + sentence[ec:]
                    texts.append(marked)
                    labels.append(label)

                enc = tok(texts, return_tensors='pt', truncation=True,
                          max_length=128, padding=True).to(self.device)
                lbl = torch.tensor(labels, dtype=torch.long).to(self.device)

                optimizer.zero_grad()
                logits = model(**enc).logits
                loss   = criterion(logits, lbl)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            correct = 0
            with torch.no_grad():
                for sentence, word, sc, ec, label in val_p:
                    marked = sentence[:sc] + '§' + word + '§' + sentence[ec:]
                    enc    = tok(marked, return_tensors='pt', truncation=True,
                                 max_length=128, padding=True).to(self.device)
                    logits = model(**enc).logits
                    pred   = logits.argmax(-1).item()
                    if pred == label:
                        correct += 1
            val_acc = correct / len(val_p) if val_p else 0.0
            print(f"  Epoch {epoch}/{epochs}  "
                  f"train_loss={total_loss/len(train_p):.4f}  "
                  f"val_acc={val_acc:.3f}")

        # Store trained model back into classifier
        self.clf.classifier = model
        model.eval()
        print("[BERTCWITrainer] Training complete.")

    def save(self, path: str = 'cwi_bert.pt') -> None:
        if self.clf.classifier is None:
            print("[BERTCWITrainer] Nothing to save (classifier is None).")
            return
        torch.save(self.clf.classifier.state_dict(), path)
        print(f"[BERTCWITrainer] Saved fine-tuned weights to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SBERT Semantic Drift Signal
# ─────────────────────────────────────────────────────────────────────────────

class SBERTDriftScorer:
    """
    Uses Sentence-BERT (all-mpnet-base-v2) to measure how far a word's
    contextual usage drifts from a neutral, concrete reference sentence.

    Reference template: "I see the [word]."
        - A concrete, literal usage.
        - High cosine distance from this template = abstract / figurative.

    Drift = (1 - cosine_sim(context_emb, template_emb)) / 2.0  ∈ [0, 1]
    (Dividing by 2 normalises from [-1,1] cosine space to [0,1].)
    """

    MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'

    def __init__(self, device=None) -> None:
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._model  = None
        self._loaded = False
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model  = SentenceTransformer(self.MODEL_NAME)
            self._model.to(self.device)
            self._loaded = True
            print(f"[SBERTDriftScorer] Loaded {self.MODEL_NAME}")
        except ImportError:
            print("[SBERTDriftScorer] sentence-transformers not installed. "
                  "Install with: pip install sentence-transformers. "
                  "Drift signal will return 0.5 (neutral).")
        except Exception as e:
            print(f"[SBERTDriftScorer] Could not load model: {e}. "
                  "Drift signal will return 0.5.")

    def _encode(self, text: str) -> np.ndarray:
        """Returns a normalised L2 sentence embedding."""
        emb = self._model.encode(text, convert_to_numpy=True,
                                 normalize_embeddings=True)
        return emb

    def compute_drift(self, sentence: str, word: str) -> float:
        """
        Returns semantic drift ∈ [0.0, 1.0].

        0.0 = word used literally (close to template)
        1.0 = word used abstractly / figuratively (far from template)
        """
        if not self._loaded or self._model is None:
            return 0.5   # neutral fallback

        template = f"I see the {word}."

        try:
            ctx_emb  = self._encode(sentence)
            tmpl_emb = self._encode(template)
            cos_sim  = float(np.dot(ctx_emb, tmpl_emb))   # L2-normed → dot = cosine
            # Map cosine ∈ [-1,1] → drift ∈ [0,1]
            drift = (1.0 - cos_sim) / 2.0
            return max(0.0, min(1.0, drift))
        except Exception:
            return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Main: DynamicContextualCWI (upgraded ensemble)
# ─────────────────────────────────────────────────────────────────────────────

class DynamicContextualCWI:
    """
    Precision-first CWI — 3-signal ensemble:

        ensemble_score = W_BERT  × DeBERTa-v3 token classifier score
                       + W_DRIFT × SBERT semantic drift
                       + W_ZIPF  × soft frequency adjustment
                       + W_STRUCT × structural pattern score
                       + structural boost (added directly if pattern detected)

    Signal 1 is now DeBERTaCWIClassifier (deberta-v3-base) instead of
    BERTCWIClassifier.  Uses score_all() for a single model forward pass
    over all content words — ~10× faster than per-word calls.

    All thresholds and weights read from CWIConfig.
    """

    def __init__(
        self,
        tokenizer=None,
        model=None,
        bert_model=None,
        device=None,
        nlp=None,
        config=None,
    ) -> None:
        from cwi_config import CWIConfig, DEFAULT_CONFIG
        self.cfg    = config if config is not None else DEFAULT_CONFIG
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Signal 1: DeBERTa-v3 token classifier (drops in over BERTCWIClassifier)
        try:
            from cwi_model import DeBERTaCWIClassifier
            self.bert_cwi = DeBERTaCWIClassifier(config=self.cfg, device=self.device)
        except Exception as e:
            print(f"[DynamicContextualCWI] DeBERTa unavailable ({e}). "
                  "Falling back to BERTCWIClassifier.")
            self.bert_cwi = BERTCWIClassifier(tokenizer, model, bert_model, device)

        # Signal 2: SBERT drift
        self.sbert_drift = SBERTDriftScorer(device)

        # Structural abstraction detector (SpaCy dep-parse)
        self.structural_detector = StructuralAbstractionDetector(nlp, self.cfg)

        # Fallback scorer for backward compat (validator Gate 4)
        self._fallback_scorer = getattr(
            self.bert_cwi, '_fallback_scorer',
            getattr(self.bert_cwi, '_fallback', None))

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers  (read from config)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def W_BERT(self):   return getattr(self.cfg, 'bert_weight', 0.40)
    @property
    def W_DRIFT(self):  return getattr(self.cfg, 'drift_weight', 0.45)
    @property
    def W_ZIPF(self):   return getattr(self.cfg, 'zipf_weight', 0.08)
    @property
    def W_STRUCT(self): return getattr(self.cfg, 'structural_weight', 0.07)
    @property
    def DRIFT_OVERRIDE_THRESHOLD(self): return getattr(self.cfg, 'drift_alone_override', 0.45)
    @property
    def K_STD(self):                    return getattr(self.cfg, 'dynamic_k_standard', 0.6)
    @property
    def MIN_DYNAMIC_THRESHOLD(self):    return getattr(self.cfg, 'min_threshold', 0.30)
    @property
    def SHORT_SENTENCE_LIMIT(self):     return self.cfg.short_sentence_limit
    @property
    def FALLBACK_SHORT_THRESH(self):    return self.cfg.short_sentence_threshold
    @property
    def COMMON_WORD_ZIPF_CEIL(self):    return getattr(self.cfg, 'zipf_always_simple', 6.5)

    def _zipf_soft_penalty(self, zipf: float, pattern_detected: bool = False) -> float:
        """
        Soft frequency penalty (replaces hard ceiling).
        Returns a score penalty in [0, cfg.max_freq_penalty].
        Common abstract words (nature, spirit) at Zipf 5.0 get a tiny penalty,
        not an outright SIMPLE override.
        """
        if pattern_detected and getattr(self.cfg, 'disable_penalty_if_pattern', True):
            return 0.0
        floor = getattr(self.cfg, 'freq_simple_floor', 4.0)
        max_penalty = getattr(self.cfg, 'max_freq_penalty', 0.02)
        if zipf <= floor:
            return 0.0
        penalty = (zipf - floor) * 0.01
        return min(max_penalty, penalty)

    def _ensemble_score(
        self,
        bert_score:  float,
        drift_score: float,
        zipf:        float,
        structural_score: float,
        pattern_boost: float,
        zipf_penalty: float,
    ) -> float:
        """Weighted ensemble with soft Zipf penalty. Result clamped to [0, 1]."""
        # Map zipf ∈ [2.0, 5.0] to score ∈ [1.0, 0.0]
        # Zipf ≤ 2.0 → 1.0 (rare/complex), Zipf ≥ 5.0 → 0.0 (common/simple)
        zipf_sig = max(0.0, min(1.0, (5.0 - zipf) / 3.0))
        raw = (self.W_BERT  * bert_score
             + self.W_DRIFT * drift_score
             + self.W_ZIPF  * zipf_sig
             + self.W_STRUCT * structural_score)
        raw += pattern_boost
        raw -= zipf_penalty   # soft penalty, never zeroes the score
        return max(0.0, min(1.0, raw))

    def _effective_threshold(
        self, mean: float, std: float, n: int, has_figurative: bool
    ) -> float:
        """Per-sentence dynamic threshold."""
        if n < self.SHORT_SENTENCE_LIMIT:
            return self.FALLBACK_SHORT_THRESH
        k = getattr(self.cfg, 'dynamic_k_figurative', 0.4) if has_figurative else getattr(self.cfg, 'dynamic_k_standard', 0.6)
        thresh = mean + k * std
        return max(self.MIN_DYNAMIC_THRESHOLD, min(getattr(self.cfg, 'max_threshold', 0.65), thresh))

    # ─────────────────────────────────────────────────────────────────────────
    # Public API (drop-in replacement for old DynamicContextualCWI)
    # ─────────────────────────────────────────────────────────────────────────

    def identify_complex_words(
        self,
        sentence:       str,
        content_tokens: List[Tuple[str, str, int, int]],
        cwi_threshold:  float = 0.35,   # legacy param, kept for API compat
    ) -> List[Dict[str, Any]]:
        """
        Identify complex words using the 3-signal ensemble.

        UPGRADED: uses DeBERTaCWIClassifier.score_all() to score ALL content
        words in ONE forward pass — much faster than per-word BERT calls.

        Hard Zipf ceiling REMOVED. Replaced with soft penalty that does not
        zero-out common abstract words (nature, spirit at Zipf ~5.0).

        Parameters
        ----------
        content_tokens : [(word, pos, start_char, end_char), ...]
        cwi_threshold  : ignored (overridden by dynamic threshold).

        Returns
        -------
        List of dicts with keys:
            word, pos, start_char, end_char,
            bert_score, drift_score, zipf_penalty, ensemble_score, final_score,
            word_zipf, is_figurative, is_complex,
            effective_threshold, threshold, mean, std, score, adjusted_score,
            pattern_detected, structural_score, pattern_boost
        """
        if not content_tokens:
            return []

        results:         List[Dict[str, Any]] = []
        ensemble_scores: List[float]          = []

        # ── Step 1: Score ALL words with DeBERTa in ONE pass ─────────────────
        # score_all() returns {word: P(complex)} for all content tokens
        try:
            bert_scores: Dict[str, float] = self.bert_cwi.score_all(
                sentence, content_tokens)
        except Exception:
            # Fallback: score one at a time
            bert_scores = {}
            for word, pos, sc, ec in content_tokens:
                try:
                    bert_scores[word] = self.bert_cwi.score(
                        sentence, word, sc, ec, pos)
                except Exception:
                    bert_scores[word] = 0.5

        # Get spaCy Doc for dependency parsing
        doc = None
        if self.structural_detector.nlp is not None:
            doc = self.structural_detector.nlp(sentence)

        for word, pos, start_char, end_char in content_tokens:
            word_zipf   = wordfreq.zipf_frequency(word.lower(), 'en')
            bert_s      = bert_scores.get(word, 0.5)
            drift_s     = self.sbert_drift.compute_drift(sentence, word)
            
            target_token = None
            if doc is not None:
                for token in doc:
                    if token.idx <= start_char < token.idx + len(token.text):
                        target_token = token
                        break

            pattern_detected, structural_score, pattern_boost = self.structural_detector.detect(doc, target_token)
            is_fig = (pattern_detected != "None")

            zipf_pen    = self._zipf_soft_penalty(word_zipf, pattern_detected=is_fig)
            ens         = self._ensemble_score(bert_s, drift_s, word_zipf, structural_score, pattern_boost, zipf_pen)

            results.append({
                'word': word, 'pos': pos,
                'start_char': start_char, 'end_char': end_char,
                'bert_score': bert_s, 'drift_score': drift_s,
                'zipf_penalty': zipf_pen,
                # legacy alias
                'zipf_adj': -zipf_pen,
                'ensemble_score': ens,
                'final_score': ens,
                'word_zipf': word_zipf,
                'is_figurative': is_fig,
                'pattern_detected': pattern_detected,
                'structural_score': structural_score,
                'pattern_boost': pattern_boost,
            })
            ensemble_scores.append(ens)

        # ── Step 2: Sentence-level statistics ────────────────────────────────
        arr        = np.array(ensemble_scores)
        mean_score = float(np.mean(arr)) if len(arr) > 0 else 0.0
        std_score  = float(np.std(arr)) if len(arr) > 0 else 0.0
        n_words    = len(content_tokens)

        figurative_pattern_count = sum(1 for res in results if res['pattern_detected'] != "None")
        has_figurative = (figurative_pattern_count > 0)
        eff_thresh = self._effective_threshold(mean_score, std_score, n_words, has_figurative)

        # ── Step 3: Classify each word ────────────────────────────────────────
        for res in results:
            res['mean']                = mean_score
            res['std']                 = std_score
            res['effective_threshold'] = eff_thresh
            res['threshold']           = eff_thresh

            bert_s  = res['bert_score']
            drift_s = res['drift_score']
            ens     = res['ensemble_score']
            word_zipf = res['word_zipf']
            pattern_detected = res['pattern_detected']
            pos = res['pos']

            # Apply override rules:
            # Rule 1: drift > drift_alone_override -> complex
            # Rule 2: drift > drift_pattern_override AND structural_pattern -> complex
            # Rule 3: zipf < zipf_always_complex -> complex
            # Rule 4: zipf > zipf_always_simple AND bert < bert_simple_threshold -> simple
            # Rule 5: Precision-first bypasses for rare nouns/adjectives/verbs
            
            is_complex = False
            if word_zipf >= getattr(self.cfg, 'zipf_always_simple', 5.2):
                is_complex = False
            elif drift_s > getattr(self.cfg, 'drift_alone_override', 0.42) and word_zipf < 4.3:
                is_complex = True
            elif drift_s > getattr(self.cfg, 'drift_pattern_override', 0.35) and pattern_detected != "None":
                is_complex = True
            elif word_zipf < getattr(self.cfg, 'zipf_always_complex', 2.5):
                is_complex = True
            elif pos in ('NOUN', 'ADJ', 'PROPN') and word_zipf < 4.3 and bert_s > 0.58:
                is_complex = True
            elif pos == 'VERB' and word_zipf < 4.3 and bert_s > 0.65:
                is_complex = True
            else:
                is_complex = ens >= eff_thresh

            res['is_complex'] = is_complex

            # Legacy field aliases for downstream compatibility
            res['score']          = ens
            res['adjusted_score'] = ens

        return results

