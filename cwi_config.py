# cwi_config.py
"""
Central configuration for the entire CWI system.
All tunable parameters live here — no magic numbers in other files.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CWIConfig:
    # ── Model ─────────────────────────────────────────────────────────────────
    deberta_model_name: str     = "microsoft/deberta-v3-base"
    sbert_model_name:   str     = "sentence-transformers/all-mpnet-base-v2"
    max_seq_length:     int     = 256
    weights_path:       str     = "cwi_bert_figurative_v2.pt"

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:         int     = 16          # reduce to 8 if OOM on GPU
    learning_rate:      float   = 2e-5        # DeBERTa standard
    weight_decay:       float   = 0.01
    warmup_ratio:       float   = 0.10        # 10% of steps for LR warmup
    num_epochs:         int     = 4
    early_stopping_patience: int = 2          # epochs to wait for precision gain
    val_split:          float   = 0.10        # 10% of CWI 2018 for validation
    test_split:         float   = 0.10        # 10% for final test

    # ── Focal Loss ─────────────────────────────────────────────────────────────
    focal_gamma:        float   = 2.0         # down-weight easy examples
    focal_alpha:        float   = 0.75        # up-weight minority class (complex)

    # ── Data ───────────────────────────────────────────────────────────────────
    cwi_2018_train_dir: str     = "cwi2018_data"   # directory of CWI 2018 files
    lex_mturk_path:     str     = "lex_mturk.txt"
    benchls_path:       str     = "BenchLS.txt"

    # ── Inference / Ensemble ──────────────────────────────────────────────────
    # New ensemble weights (total = 1.0)
    bert_weight:        float   = 0.40
    drift_weight:       float   = 0.45
    zipf_weight:        float   = 0.08
    structural_weight:  float   = 0.07

    # Backward compatibility properties
    @property
    def ensemble_deberta_weight(self) -> float:
        return self.bert_weight

    @property
    def ensemble_sbert_weight(self) -> float:
        return self.drift_weight

    # Override thresholds
    drift_pattern_override: float = 0.35
    drift_alone_override:   float = 0.42
    zipf_always_complex:    float = 2.5
    zipf_always_simple:     float = 5.0
    bert_simple_threshold:  float = 0.40

    # Dynamic threshold
    dynamic_k_figurative:   float = 0.4
    dynamic_k_standard:     float = 0.4
    min_threshold:          float = 0.32
    max_threshold:          float = 0.65

    # Frequency adjustment
    freq_simple_floor:      float = 4.0
    max_freq_bonus:         float = 0.08
    max_freq_penalty:       float = 0.02
    disable_penalty_if_pattern: bool = True

    # Structural detection boosts
    structural_boost_pattern_a: float = 0.15
    structural_boost_pattern_b: float = 0.10
    structural_boost_pattern_c: float = 0.12

    # Legacy configuration options kept for safety/fallback
    k_std:                  float = 0.5
    threshold_floor:        float = 0.30
    short_sentence_limit:   int   = 6
    short_sentence_threshold: float = 0.35
    drift_override_threshold: float = 0.42
    zipf_hard_ceiling:      float = 5.5

    # ── Logging ────────────────────────────────────────────────────────────────
    verbose: bool = True

    def describe(self) -> str:
        lines = [
            "=== CWI Configuration ===",
            f"  Model:     {self.deberta_model_name}",
            f"  Weights:   {self.weights_path}",
            f"  Focal:     gamma={self.focal_gamma}  alpha={self.focal_alpha}",
            f"  Ensemble:  BERT={self.bert_weight} Drift={self.drift_weight} Zipf={self.zipf_weight} Structural={self.structural_weight}",
            f"  Overrides: drift_pattern_override={self.drift_pattern_override} drift_alone_override={self.drift_alone_override}",
            f"  Dynamic:   k_standard={self.dynamic_k_standard} k_figurative={self.dynamic_k_figurative} bounds=[{self.min_threshold}, {self.max_threshold}]",
        ]
        return "\n".join(lines)


# Default singleton used by all modules unless overridden
DEFAULT_CONFIG = CWIConfig()

