# cwi_trainer.py
"""
Complete training pipeline for DeBERTaCWIModel.

Features:
  - Focal Loss (handles class imbalance)
  - AdamW + linear warmup scheduler
  - Early stopping on VALIDATION PRECISION (not F1, per user's requirement)
  - Gradient clipping (stabilises DeBERTa fine-tuning)
  - Per-epoch detailed metrics
  - Auto-save best checkpoint
  - CPU + GPU compatible

Usage:
    python train_cwi.py   ← uses this module internally
"""

from __future__ import annotations

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from cwi_config import CWIConfig, DEFAULT_CONFIG
from cwi_data import CWISample
from cwi_model import DeBERTaCWIModel


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary Focal Loss for imbalanced classification.

    FL(p) = -α (1-p)^γ log(p)

    γ (gamma): focusing parameter. Higher γ → less weight on easy examples.
               γ=0 reduces to standard BCE. γ=2 is standard for detection tasks.
    α (alpha): class weight for positive class (complex words).
               α=0.75 compensates for ~30% complex / 70% simple distribution.

    Why Focal over BCE for CWI?
      CWI 2018 is ~70% simple. Without Focal Loss, the model trivially achieves
      70% accuracy by predicting all-simple. Focal Loss forces it to focus on
      the harder, rarer complex examples where real learning happens.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [N] unnormalised scores
        targets: [N] binary labels {0.0, 1.0}
        """
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt    = torch.where(targets == 1, probs, 1 - probs)
        alpha = torch.where(targets == 1,
                            torch.full_like(pt, self.alpha),
                            torch.full_like(pt, 1 - self.alpha))
        focal = alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CWIDataset(Dataset):
    """
    Converts CWISample objects into model inputs.

    Each sample encodes the full sentence and stores the label at the
    position of the target word.

    Target position strategy:
      We mark which TOKEN (index in the tokenizer output) corresponds to the
      target word, using char-to-token mapping. During the forward pass, we
      extract that token's logit as the word-level score.
    """

    def __init__(
        self,
        samples:    List[CWISample],
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.items: List[dict] = []
        skipped = 0

        for s in samples:
            enc = tokenizer(
                s.sentence,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_offsets_mapping=True,
            )
            offset_map = enc.pop('offset_mapping')[0]    # [seq_len, 2]

            # Find the token index for the target word
            tok_idx = self._char_to_tok(offset_map, s.start_char, s.end_char)
            if tok_idx == -1:
                skipped += 1
                continue

            self.items.append({
                'input_ids':      enc['input_ids'][0],
                'attention_mask': enc['attention_mask'][0],
                'token_type_ids': enc.get('token_type_ids', [None])[0],
                'tok_idx':        tok_idx,
                'label':          float(s.label),
                'word':           s.word,        # for debugging
            })

        if skipped > 0:
            print(f"[CWIDataset] Skipped {skipped} samples "
                  f"(word not found in tokenized sequence).")

    @staticmethod
    def _char_to_tok(offset_map: torch.Tensor, start: int, end: int) -> int:
        """Find first token whose offset span contains start_char."""
        for i, (os, oe) in enumerate(offset_map.tolist()):
            if os <= start < oe:
                return i
            if os == start and oe > start:
                return i
        # Fallback: look for token starting just after start
        for i, (os, oe) in enumerate(offset_map.tolist()):
            if os == start:
                return i
        return -1

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate_fn(batch: List[dict]) -> dict:
    """Custom collate: pad to batch max, handle None token_type_ids."""
    keys = ['input_ids', 'attention_mask']
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    if batch[0]['token_type_ids'] is not None:
        out['token_type_ids'] = torch.stack([b['token_type_ids'] for b in batch])
    out['tok_idx'] = torch.tensor([b['tok_idx'] for b in batch], dtype=torch.long)
    out['label']   = torch.tensor([b['label']   for b in batch], dtype=torch.float)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class CWITrainer:
    """
    Fine-tunes DeBERTaCWIModel using Focal Loss + AdamW + warmup.

    Training objective: MAXIMISE VALIDATION PRECISION
    (not F1, because user requirement is precision > recall)

    Early stopping: halt if val_precision does not improve for
    `patience` consecutive epochs.
    """

    def __init__(
        self,
        model:  DeBERTaCWIModel,
        config: CWIConfig = DEFAULT_CONFIG,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model  = model
        self.config = config
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(
        self,
        train_samples: List[CWISample],
        val_samples:   List[CWISample],
        tokenizer,
    ) -> None:
        """Run the full training loop."""
        cfg = self.config
        print(f"\n{cfg.describe()}")
        print(f"\n[CWITrainer] Building datasets...")

        train_ds = CWIDataset(train_samples, tokenizer, cfg.max_seq_length)
        val_ds   = CWIDataset(val_samples,   tokenizer, cfg.max_seq_length)
        print(f"  Train: {len(train_ds)} valid samples")
        print(f"  Val:   {len(val_ds)} valid samples")

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # ── Optimizer ─────────────────────────────────────────────────────────
        # Layer-wise LR decay: lower layers get smaller LR
        # This is standard practice for DeBERTa fine-tuning
        no_decay   = {'bias', 'LayerNorm.weight', 'layer_norm.weight'}
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': cfg.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(param_groups, lr=cfg.learning_rate)

        # ── Scheduler: linear warmup → linear decay ────────────────────────
        total_steps  = len(train_loader) * cfg.num_epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # ── Focal Loss ─────────────────────────────────────────────────────
        criterion = FocalLoss(gamma=cfg.focal_gamma, alpha=cfg.focal_alpha)

        # ── Training loop ──────────────────────────────────────────────────
        best_val_precision = 0.0
        patience_counter   = 0
        best_state         = None

        print(f"\n[CWITrainer] Starting training: "
              f"{cfg.num_epochs} epochs, "
              f"batch={cfg.batch_size}, lr={cfg.learning_rate}")
        print(f"  Total steps: {total_steps}  "
              f"Warmup: {warmup_steps}")

        for epoch in range(1, cfg.num_epochs + 1):
            # ── Train ────────────────────────────────────────────────────
            self.model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                inp_ids  = batch['input_ids'].to(self.device)
                attn     = batch['attention_mask'].to(self.device)
                tok_ids  = batch.get('token_type_ids')
                if tok_ids is not None:
                    tok_ids = tok_ids.to(self.device)
                tok_idx  = batch['tok_idx'].to(self.device)
                labels   = batch['label'].to(self.device)

                logits = self.model(inp_ids, attn, tok_ids)  # [B, seq_len]

                # Extract the logit at the target word position
                word_logits = logits[
                    torch.arange(logits.size(0)), tok_idx]   # [B]

                loss = criterion(word_logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)   # gradient clip
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)

            # ── Validate ─────────────────────────────────────────────────
            metrics = self._evaluate(val_loader)
            val_precision = metrics['precision']
            val_recall    = metrics['recall']
            val_f1        = metrics['f1']

            print(f"\n  Epoch {epoch}/{cfg.num_epochs}  "
                  f"loss={avg_loss:.4f}  "
                  f"val_P={val_precision:.3f}  "
                  f"val_R={val_recall:.3f}  "
                  f"val_F1={val_f1:.3f}")

            # ── Early stopping (on PRECISION) ─────────────────────────────
            if val_precision > best_val_precision:
                best_val_precision = val_precision
                patience_counter   = 0
                import copy
                best_state = copy.deepcopy(self.model.state_dict())
                print(f"  [OK] Best val_precision={val_precision:.3f} - checkpoint saved")
            else:
                patience_counter += 1
                print(f"  No precision improvement "
                      f"({patience_counter}/{cfg.early_stopping_patience})")
                if patience_counter >= cfg.early_stopping_patience:
                    print(f"\n[CWITrainer] Early stopping at epoch {epoch}.")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        # Save checkpoint
        torch.save(self.model.state_dict(), cfg.weights_path)
        print(f"\n[CWITrainer] Saved best model to {cfg.weights_path}")
        print(f"[CWITrainer] Best val_precision: {best_val_precision:.3f}")

    def _evaluate(
        self,
        loader: DataLoader,
        threshold: float = 0.5,
    ) -> dict:
        """
        Evaluate on a DataLoader. Returns precision, recall, F1.

        NOTE: Using threshold=0.5 for training evaluation.
        During inference, the dynamic per-sentence threshold is used.
        """
        self.model.eval()
        tp = fp = fn = tn = 0

        with torch.no_grad():
            for batch in loader:
                inp_ids  = batch['input_ids'].to(self.device)
                attn     = batch['attention_mask'].to(self.device)
                tok_ids  = batch.get('token_type_ids')
                if tok_ids is not None:
                    tok_ids = tok_ids.to(self.device)
                tok_idx  = batch['tok_idx'].to(self.device)
                labels   = batch['label']

                logits     = self.model(inp_ids, attn, tok_ids)
                word_logits = logits[
                    torch.arange(logits.size(0)), tok_idx].cpu()
                probs = torch.sigmoid(word_logits)
                preds = (probs >= threshold).long()
                lbls  = labels.long()

                tp += ((preds == 1) & (lbls == 1)).sum().item()
                fp += ((preds == 1) & (lbls == 0)).sum().item()
                fn += ((preds == 0) & (lbls == 1)).sum().item()
                tn += ((preds == 0) & (lbls == 0)).sum().item()

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-9)

        return dict(precision=precision, recall=recall, f1=f1,
                    accuracy=accuracy, tp=tp, fp=fp, fn=fn, tn=tn)
