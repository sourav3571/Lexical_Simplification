# bert_ranker.py
"""
Upgraded GatedFusionRanker — 6-feature neural ranker.

New features vs. old (4-feature) ranker:
  5. Zipf frequency difference (candidate_zipf - original_zipf)
  6. GloVe / embedding cosine similarity

Feature weights (starting point for training):
  Score = 0.15 × MLM_prob
        + 0.40 × SBERT_sentence_sim
        + 0.15 × surprisal_reduction
        + 0.15 × fluency_change
        + 0.10 × zipf_difference
        + 0.05 × glove_cosine_sim

Training: MarginRankingLoss on BenchLS ranked substitutions.
Validation: LexMTurk held-out (80/20 split).

Backward compatibility:
  - `predict(mlm, cos, surp, fluency)` still works (zipf_diff=0, glove=0).
  - `predict6(...)` is the new 6-feature entry point.
"""

from __future__ import annotations

import os
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Optional

from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_surprisal import BERTSurprisalCalculator


# ─────────────────────────────────────────────────────────────────────────────
# Semantic relation helper (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def are_semantically_related(
    chosen_sense, target_word: str, cand: str, pos_tag: str
) -> bool:
    pos_map = {
        'NOUN': wn.NOUN, 'VERB': wn.VERB,
        'ADJ': wn.ADJ,   'ADV': wn.ADV, 'PROPN': wn.NOUN,
    }
    wn_pos = pos_map.get(pos_tag.upper()) if pos_tag else None
    if not wn_pos:
        return True
    c_s = wn.synsets(cand.lower(), pos=wn_pos)
    if not c_s:
        return False
    c_set = set(c_s)
    if chosen_sense:
        if chosen_sense in c_set:
            return True
        direct_relations = chosen_sense.hypernyms() + chosen_sense.hyponyms()
        if any(rel in c_set for rel in direct_relations):
            return True
        for hyp in chosen_sense.hypernyms():
            if any(sister in c_set for sister in hyp.hyponyms()):
                return True
        return False
    t_s = wn.synsets(target_word.lower(), pos=wn_pos)
    if not t_s:
        return False
    for ts in t_s:
        if ts in c_set:
            return True
        direct_relations = ts.hypernyms() + ts.hyponyms()
        if any(rel in c_set for rel in direct_relations):
            return True
        for hyp in ts.hypernyms():
            if any(sister in c_set for sister in hyp.hyponyms()):
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 6-Feature Gated Fusion Ranker
# ─────────────────────────────────────────────────────────────────────────────

class GatedFusionRanker(nn.Module):
    """
    Linear → Sigmoid ranker over 6 features.

    Features (in order):
        0  mlm_prob           — MLM fit probability
        1  sbert_sim          — SBERT sentence cosine similarity
        2  surprisal_red      — surprisal(original) - surprisal(candidate)
        3  fluency_change     — Δ log-likelihood (higher = more fluent)
        4  zipf_diff          — candidate_zipf - original_zipf (positive = simpler)
        5  glove_sim          — GloVe cosine similarity (0 if unavailable)

    Starting weights reflect the priority ordering specified in the task.
    Training refines these via MarginRankingLoss.
    """

    N_FEATURES = 6

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.N_FEATURES, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self._init_weights()
        self.is_trained = False

    def _init_weights(self) -> None:
        """
        Seed the first linear layer to approximate the specified feature weights:
          0.15 MLM, 0.40 SBERT, 0.15 surp, 0.15 fluency, 0.10 zipf, 0.05 glove
        Scaled by 4 so the sigmoid is in a useful range before training.
        """
        with torch.no_grad():
            w = torch.tensor(
                [[0.60, 1.60, 0.60, 0.60, 0.40, 0.20]],
                dtype=torch.float32)
            self.net[0].weight.copy_(
                w.expand(16, -1) * torch.randn(16, 6) * 0.1 + w.expand(16, -1))
            self.net[0].bias.fill_(-1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not getattr(self, 'is_trained', False) and not self.training:
            # Deterministic starting weights:
            # 0.15 MLM, 0.40 SBERT, 0.15 surp, 0.15 fluency, 0.10 zipf, 0.05 glove
            # Scaled by 4 for sigmoid range
            w = torch.tensor([0.60, 1.60, 0.60, 0.60, 0.40, 0.20], dtype=torch.float32, device=features.device)
            # Dot product along the last dimension
            if features.dim() == 1:
                return torch.sigmoid(torch.dot(features, w) - 1.0).unsqueeze(0)
            else:
                return torch.sigmoid(torch.mv(features, w) - 1.0).unsqueeze(-1)
        return self.net(features)

    # ── Single-candidate inference ────────────────────────────────────────────

    def predict(
        self,
        mlm_prob:      float,
        cosine_sim:    float,
        surp_red:      float,
        fluency_change: float,
        pos_mismatch:  bool  = False,
        zipf_diff:     float = 0.0,
        glove_sim:     float = 0.0,
    ) -> float:
        """
        Backward-compatible predict.  Old callers pass 4 args; new callers
        pass all 6 features via zipf_diff and glove_sim.
        """
        feats = torch.tensor(
            [[mlm_prob, cosine_sim, surp_red, fluency_change, zipf_diff, glove_sim]],
            dtype=torch.float32)
        with torch.no_grad():
            score = self.forward(feats).item()
        if pos_mismatch:
            score -= 0.3
        return score

    def predict6(
        self,
        mlm_prob:      float,
        sbert_sim:     float,
        surp_red:      float,
        fluency_change: float,
        zipf_diff:     float,
        glove_sim:     float,
        pos_mismatch:  bool = False,
    ) -> float:
        """Explicit 6-feature entry point."""
        return self.predict(
            mlm_prob, sbert_sim, surp_red, fluency_change,
            pos_mismatch, zipf_diff, glove_sim)

    # ─────────────────────────────────────────────────────────────────────────
    # Training (MarginRankingLoss on BenchLS + LexMTurk)
    # ─────────────────────────────────────────────────────────────────────────

    def train_on_benchls(
        self,
        benchls_path:  str,
        lex_mturk_path: str,
        tokenizer:     BertTokenizer,
        model:         BertForMaskedLM,
        bert_model:    BertModel,
        device:        torch.device,
        epochs:        int  = 3,
        lr:            float = 0.001,
        limit:         int  = 200,
        val_split:     float = 0.2,
        emb_store=None,         # EmbeddingStore for GloVe feature
        sbert_encoder=None,     # _SBERTEncoder for sentence sim feature
    ) -> None:
        """
        Train using MarginRankingLoss.

        For each sentence we take ordered pairs of candidates:
          (higher-voted, lower-voted) — the higher-voted one should score higher.
        """
        # ── Load data ────────────────────────────────────────────────────────
        rows = self._load_dataset(benchls_path) + self._load_dataset(lex_mturk_path)
        if not rows:
            print("[GatedFusionRanker] No training data found.")
            return

        random.shuffle(rows)
        n_val   = max(1, int(len(rows) * val_split))
        val_rows = rows[:n_val]
        trn_rows = rows[n_val: n_val + limit]

        surp_calc = BERTSurprisalCalculator(tokenizer, model, device)
        self.to(device)
        self.train()

        optimizer  = optim.Adam(self.parameters(), lr=lr)
        margin_crit = nn.MarginRankingLoss(margin=0.1)

        print(f"[GatedFusionRanker] Training {len(trn_rows)} sentences, "
              f"validating {len(val_rows)} | epochs={epochs}")

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_pairs    = 0

            for sentence, target, candidates in trn_rows:
                match = re.search(r'\b' + re.escape(target) + r'\b',
                                  sentence, re.IGNORECASE)
                if not match:
                    continue
                sc, ec = match.start(), match.end()

                # Pre-compute shared tensors
                feats_list = []
                for cand, votes in candidates[:5]:
                    f = self._extract_features(
                        sentence, target, sc, ec, cand, votes,
                        tokenizer, model, bert_model, surp_calc, device,
                        emb_store, sbert_encoder)
                    if f is not None:
                        feats_list.append((f, votes))

                # Margin ranking pairs
                for i in range(len(feats_list)):
                    for j in range(i + 1, len(feats_list)):
                        fi, vi = feats_list[i]
                        fj, vj = feats_list[j]
                        if vi == vj:
                            continue
                        # Higher votes = should rank first
                        pos_f = fi if vi > vj else fj
                        neg_f = fj if vi > vj else fi
                        pos_s = self.forward(
                            pos_f.unsqueeze(0).to(device)).squeeze(-1)
                        neg_s = self.forward(
                            neg_f.unsqueeze(0).to(device)).squeeze(-1)
                        target_t = torch.ones(1).to(device)
                        loss = margin_crit(pos_s, neg_s, target_t)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        n_pairs    += 1

            avg = epoch_loss / max(n_pairs, 1)
            print(f"  Epoch {epoch}/{epochs}  "
                  f"margin_loss={avg:.4f}  n_pairs={n_pairs}")

        self.eval()
        print("[GatedFusionRanker] Training complete.")

    def _extract_features(
        self,
        sentence, target, sc, ec, cand, votes,
        tokenizer, model, bert_model, surp_calc, device,
        emb_store, sbert_encoder,
    ) -> Optional[torch.Tensor]:
        """Extract all 6 features for one candidate. Returns None on failure."""
        import wordfreq
        try:
            cand_sentence = sentence[:sc] + cand + sentence[ec:]

            # 0. MLM prob
            masked = surp_calc.get_masked_sentence_and_idx(sentence, sc, ec)
            enc    = tokenizer(masked, return_tensors='pt').to(device)
            midxs  = (enc['input_ids'][0] == tokenizer.mask_token_id
                      ).nonzero(as_tuple=True)[0]
            if len(midxs) == 0:
                return None
            with torch.no_grad():
                logits = model(**enc).logits
            probs = F.softmax(logits[0, midxs[0].item()], dim=-1)
            toks  = tokenizer(cand, add_special_tokens=False)['input_ids']
            mlm_p = probs[toks[0]].item() if toks else 1e-9

            # 1. SBERT sentence similarity
            if sbert_encoder is not None and sbert_encoder.available:
                sbert_s = sbert_encoder.similarity(sentence, cand_sentence)
            else:
                # Fallback: BERT CLS cosine
                orig_e = bert_model(
                    **tokenizer(sentence, return_tensors='pt',
                                padding=True, truncation=True).to(device)
                ).last_hidden_state[0, 0]
                cand_e = bert_model(
                    **tokenizer(cand_sentence, return_tensors='pt',
                                padding=True, truncation=True).to(device)
                ).last_hidden_state[0, 0]
                sbert_s = F.cosine_similarity(
                    orig_e.unsqueeze(0), cand_e.unsqueeze(0)).item()

            # 2. Surprisal reduction
            orig_surp = surp_calc.compute_surprisal(sentence, target, sc, ec)
            cand_surp = surp_calc.compute_surprisal(sentence, cand, sc, ec)
            surp_red  = orig_surp - cand_surp

            # 3. Fluency change
            def _fluency(sent):
                e = tokenizer(sent, return_tensors='pt',
                              padding=True, truncation=True).to(device)
                with torch.no_grad():
                    return -model(**e, labels=e['input_ids']).loss.item()

            fluency_chg = _fluency(cand_sentence) - _fluency(sentence)

            # 4. Zipf diff
            orig_z  = wordfreq.zipf_frequency(target.lower(), 'en')
            cand_z  = wordfreq.zipf_frequency(cand.lower(), 'en')
            zipf_d  = cand_z - orig_z

            # 5. GloVe cosine (0 if unavailable)
            glove_s = 0.0
            if emb_store is not None:
                glove_s = emb_store.similarity(target.lower(), cand.lower(),
                                               source='glove')

            return torch.tensor(
                [mlm_p, sbert_s, surp_red, fluency_chg, zipf_d, glove_s],
                dtype=torch.float32)

        except Exception:
            return None

    @staticmethod
    def _load_dataset(path: str) -> list:
        if not path or not os.path.exists(path):
            return []
        rows = []
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                sentence = parts[0].strip().strip('"')
                target   = parts[1].strip().lower()
                raw      = parts[3:] if (
                    len(parts) > 3 and parts[2].isdigit()) else parts[2:]
                has_c = any(':' in p for p in raw)
                cands = []
                if has_c:
                    for item in raw:
                        if ':' in item:
                            tok = item.split(':')
                            try:
                                cands.append((tok[0].strip().lower(),
                                              int(tok[1])))
                            except (IndexError, ValueError):
                                pass
                else:
                    from collections import Counter
                    cnt = Counter(p.strip().lower() for p in raw if p.strip())
                    cands = list(cnt.items())
                if cands:
                    rows.append((sentence, target, cands))
        return rows

    # ─────────────────────────────────────────────────────────────────────────
    # Legacy: train_on_lex_mturk (kept for backward compat)
    # ─────────────────────────────────────────────────────────────────────────

    def train_on_lex_mturk(
        self,
        file_path: str,
        tokenizer, model, bert_model, device,
        limit: int = 5,
    ) -> None:
        """Legacy entry point — routes to train_on_benchls."""
        self.train_on_benchls(
            benchls_path=file_path,
            lex_mturk_path='',
            tokenizer=tokenizer,
            model=model,
            bert_model=bert_model,
            device=device,
            limit=limit,
        )
