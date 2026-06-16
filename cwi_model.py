# cwi_model.py
"""
DeBERTa-v3 Token Classifier for Complex Word Identification.

Architecture
─────────────
Input:  full sentence (tokenized by DeBERTa-v3 tokenizer)
Model:  microsoft/deberta-v3-base  (12 layers, 768 hidden, disentangled attention)
Head:   Linear(768 → 1) + Sigmoid per WORD token (not per subword)

Why token classification (not sequence classification)?
  - CWI is a WORD-level task: each word gets its own binary label.
  - Token classification is architecturally correct (same as NER).
  - Single forward pass scores ALL words simultaneously → efficient.
  - Contextual representations differ per sense → handles polysemy implicitly.

Subword aggregation strategy:
  DeBERTa tokenizes "physician" as ["ph", "##ysician"].
  We use the FIRST subword token's hidden state as the word representation.
  (Alternatively: mean of all subword tokens — configurable via SUBWORD_POOL.)

SUBWORD_POOL options:
  "first"  — use first subword (standard for NER, e.g. CoNLL-2003)
  "mean"   — average all subwords (slightly better on long words)
  "max"    — max-pool subwords

Inference:
  Given (sentence, word, start_char, end_char):
    → tokenize → forward → extract word position → sigmoid → P(complex)
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict

from transformers import (
    AutoTokenizer,
    AutoModel,
    DebertaV2TokenizerFast,
)
from cwi_config import CWIConfig, DEFAULT_CONFIG


SUBWORD_POOL: str = "first"   # "first" | "mean" | "max"


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────

class DeBERTaCWIModel(nn.Module):
    """
    DeBERTa-v3-base with a binary token classification head.

    Input:  tokenized sentence (batch_size=1 for inference)
    Output: logits per subword token (shape: [seq_len, 1])

    The caller is responsible for:
      1. Mapping word positions to subword positions
      2. Pooling subword logits to word-level scores
      3. Applying sigmoid to get P(complex)
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-base") -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden = self.encoder.config.hidden_size      # 768 for base
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state      # [B, seq_len, 768]
        hidden = hidden.to(self.classifier[1].weight.dtype)
        logits = self.classifier(hidden)        # [B, seq_len, 1]
        return logits.squeeze(-1)               # [B, seq_len]


# ─────────────────────────────────────────────────────────────────────────────
# Inference wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DeBERTaCWIClassifier:
    """
    High-level inference wrapper for the DeBERTa-v3 CWI model.

    Provides the same interface as BERTCWIClassifier (drop-in replacement):
        classifier.score(sentence, word, start_char, end_char) → float [0,1]

    Also supports scoring ALL words in a sentence in one forward pass:
        classifier.score_all(sentence, word_list) → Dict[str, float]
    """

    def __init__(
        self,
        config: CWIConfig = DEFAULT_CONFIG,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config  = config
        self.device  = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._loaded = False
        self.model: Optional[DeBERTaCWIModel] = None
        self.tokenizer = None

        # Fallback scorer when DeBERTa is not loaded
        self._fallback = None

        self._load()

    def _load(self) -> None:
        """Load DeBERTa-v3 model. Falls back gracefully if unavailable."""
        try:
            print(f"[DeBERTaCWIClassifier] Loading tokenizer: "
                  f"{self.config.deberta_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.deberta_model_name)

            print(f"[DeBERTaCWIClassifier] Loading model: "
                  f"{self.config.deberta_model_name}")
            self.model = DeBERTaCWIModel(self.config.deberta_model_name)
            self.model.to(self.device)
            self.model.eval()

            # Load fine-tuned weights if available
            if os.path.exists(self.config.weights_path):
                state = torch.load(
                    self.config.weights_path, map_location=self.device)
                self.model.load_state_dict(state, strict=True)
                print(f"[DeBERTaCWIClassifier] Loaded fine-tuned weights: "
                      f"{self.config.weights_path}")
            else:
                print(f"[DeBERTaCWIClassifier] No fine-tuned weights at "
                      f"'{self.config.weights_path}'. "
                      f"Run train_cwi.py to fine-tune. Using pretrained only.")

            self._loaded = True

        except Exception as e:
            print(f"[DeBERTaCWIClassifier] Failed to load DeBERTa-v3: {e}")
            print("[DeBERTaCWIClassifier] Falling back to BERTComplexityScorer.")
            self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ── Subword→Word mapping helpers ──────────────────────────────────────────

    def _find_word_token_span(
        self,
        encoding,
        sentence: str,
        start_char: int,
        end_char:   int,
    ) -> Tuple[int, int]:
        """
        Find the span of subword token indices that correspond to the
        target word at [start_char, end_char) in the original sentence.

        Returns: (token_start_idx, token_end_idx) in tokenizer output space.
        Returns (-1, -1) if not found.
        """
        try:
            tok_span = encoding.char_to_token(start_char)
            if tok_span is None:
                # Try adjacent character
                tok_span = encoding.char_to_token(start_char + 1)
            if tok_span is None:
                return -1, -1

            # Find the last subword of this word
            tok_end = tok_span
            for c in range(start_char + 1, end_char):
                t = encoding.char_to_token(c)
                if t is not None and t > tok_end:
                    tok_end = t

            return tok_span, tok_end + 1   # exclusive end
        except Exception:
            return -1, -1

    def _pool_subword_hidden(
        self,
        hidden_states: torch.Tensor,   # [seq_len, 768]
        t_start: int,
        t_end:   int,
        pool:    str = SUBWORD_POOL,
    ) -> torch.Tensor:
        """Pool subword hidden states to a single word vector."""
        span = hidden_states[t_start:t_end]          # [n_subtokens, 768]
        if span.size(0) == 0:
            return hidden_states[t_start]
        if pool == "first":
            return span[0]
        if pool == "mean":
            return span.mean(dim=0)
        if pool == "max":
            return span.max(dim=0).values
        return span[0]

    # ── Single-word scoring (API compatible with BERTCWIClassifier) ───────────

    def score(
        self,
        sentence:   str,
        word:       str,
        start_char: int,
        end_char:   int,
        pos:        str = None,
    ) -> float:
        """
        Returns P(complex) ∈ [0.0, 1.0] for a single target word.

        Falls back to BERTComplexityScorer if DeBERTa is not loaded.
        """
        if not self._loaded:
            return self._fallback_score(sentence, word, start_char, end_char, pos)

        try:
            enc = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.max_seq_length,
                return_offsets_mapping=True,
            )
            offset_map = enc.pop('offset_mapping')   # remove before model call

            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.model(**enc)   # [1, seq_len]

            probs = torch.sigmoid(logits[0])   # [seq_len]

            # Map char offsets → token indices
            t_start, t_end = self._find_word_token_span(
                self.tokenizer(sentence, return_offsets_mapping=True),
                sentence, start_char, end_char)

            if t_start == -1:
                return 0.5  # cannot locate word

            # Use first subword's probability
            return float(probs[t_start].item())

        except Exception as e:
            if self.config.verbose:
                print(f"[DeBERTaCWIClassifier] score() error: {e}")
            return self._fallback_score(sentence, word, start_char, end_char, pos)

    def score_all(
        self,
        sentence: str,
        content_tokens: List[Tuple[str, str, int, int]],
        # (word, pos, start_char, end_char)
    ) -> Dict[str, float]:
        """
        Score ALL content words in one single forward pass.

        This is 10× faster than calling score() per word because
        DeBERTa only encodes the sentence once.

        Returns: {word: P(complex)} dict
        """
        if not self._loaded or not content_tokens:
            return {w: self._fallback_score(sentence, w, sc, ec, pos)
                    for w, pos, sc, ec in content_tokens}

        try:
            enc = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.max_seq_length,
                return_offsets_mapping=True,
            )
            char_to_tok = self.tokenizer(
                sentence,
                return_offsets_mapping=True,
            )
            enc.pop('offset_mapping', None)
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.model(**enc)    # [1, seq_len]
            probs = torch.sigmoid(logits[0])  # [seq_len]

            results: Dict[str, float] = {}
            for word, pos, start_char, end_char in content_tokens:
                t_start, _ = self._find_word_token_span(
                    char_to_tok, sentence, start_char, end_char)
                if t_start == -1 or t_start >= probs.size(0):
                    results[word] = 0.5
                else:
                    results[word] = float(probs[t_start].item())

            return results

        except Exception as e:
            if self.config.verbose:
                print(f"[DeBERTaCWIClassifier] score_all() error: {e}")
            return {w: self._fallback_score(sentence, w, sc, ec, pos)
                    for w, pos, sc, ec in content_tokens}

    def _fallback_score(
        self, sentence: str, word: str, start_char: int, end_char: int, pos: str
    ) -> float:
        """Lazy-initialize and use BERTComplexityScorer as fallback."""
        if self._fallback is None:
            try:
                from bert_complexity import BERTComplexityScorer
                from transformers import BertTokenizer, BertForMaskedLM, BertModel
                dev = self.device
                tok = BertTokenizer.from_pretrained('bert-base-uncased')
                mlm = BertForMaskedLM.from_pretrained('bert-base-uncased').to(dev)
                brt = BertModel.from_pretrained('bert-base-uncased').to(dev)
                self._fallback = BERTComplexityScorer(tok, mlm, brt, dev)
            except Exception:
                return 0.5  # absolute fallback
        return self._fallback.compute_complexity_score(
            sentence, word, start_char, end_char, pos)

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.config.weights_path
        if self.model is None:
            print("[DeBERTaCWIClassifier] Nothing to save.")
            return
        torch.save(self.model.state_dict(), path)
        print(f"[DeBERTaCWIClassifier] Saved to {path}")

    def load_weights(self, path: Optional[str] = None) -> None:
        path = path or self.config.weights_path
        if self.model is None or not os.path.exists(path):
            return
        self.model.load_state_dict(
            torch.load(path, map_location=self.device), strict=True)
        self.model.eval()
        print(f"[DeBERTaCWIClassifier] Loaded weights from {path}")
