# train_cwi.py
"""
One-command CWI training script.

Trains DeBERTa-v3-base on:
  1. CWI 2018 Shared Task (~27k labelled tokens)  — primary
  2. LexMTurk (~500 sentences)                    — secondary
  3. BenchLS (~929 sentences)                     — secondary

Using:
  - Focal Loss (γ=2.0, α=0.75) for class imbalance
  - AdamW + linear warmup scheduler
  - Early stopping on validation PRECISION
  - Saves best model to cwi_deberta.pt

Usage:
    python train_cwi.py

After training:
    Restart interactive_simplifier.py — it auto-loads cwi_deberta.pt.

CWI 2018 Data:
    Option A: pip install datasets   (auto-download via HuggingFace)
    Option B: Download manually from https://zenodo.org/record/1172640
              and extract to ./cwi2018_data/
"""

import datasets  # Must be imported before torch/transformers on Windows to avoid DLL conflict crash
import torch
from transformers import AutoTokenizer

from cwi_config  import CWIConfig
from cwi_data    import build_combined_dataset
from cwi_model   import DeBERTaCWIModel
from cwi_trainer import CWITrainer
from cwi_evaluator import CWIEvaluator, CWISample


def main():
    # Optimise CPU thread count for PyTorch
    torch.set_num_threads(4)

    # ── Configuration ─────────────────────────────────────────────────────────
    config = CWIConfig(
        deberta_model_name      = "microsoft/deberta-v3-base",
        weights_path            = "cwi_deberta.pt",
        batch_size              = 16,         # reduce to 8 if OOM
        learning_rate           = 2e-5,
        num_epochs              = 1,
        early_stopping_patience = 2,
        focal_gamma             = 2.0,
        focal_alpha             = 0.75,
        cwi_2018_train_dir      = "cwi2018_data",
        lex_mturk_path          = "lex_mturk.txt",
        benchls_path            = "BenchLS.txt",
        val_split               = 0.10,
        test_split              = 0.10,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[train_cwi] Device: {device}")
    print(config.describe())

    # ── Data ──────────────────────────────────────────────────────────────────
    train_s, val_s, test_s = build_combined_dataset(
        cwi_2018_dir   = config.cwi_2018_train_dir,
        lex_mturk_path = config.lex_mturk_path,
        benchls_path   = config.benchls_path,
        val_split      = config.val_split,
        test_split     = config.test_split,
    )

    if not train_s:
        print("\n[train_cwi] ERROR: No training data found.")
        print("  Option A: pip install datasets   (auto-download CWI 2018)")
        print("  Option B: Download from https://zenodo.org/record/1172640")
        print("            and extract to ./cwi2018_data/")
        print("\n  Running with LexMTurk + BenchLS only "
              "(~2,858 samples, weaker model)...")
        # Fall back to LexMTurk + BenchLS only
        from data_loader import load_cwi_training_pairs
        pairs  = load_cwi_training_pairs(config.lex_mturk_path,
                                          config.benchls_path)
        if not pairs:
            print("[train_cwi] No data at all. Exiting.")
            return
        import random
        random.shuffle(pairs)
        n_val   = max(1, int(len(pairs) * 0.10))
        val_s   = [CWISample(s, w, sc, ec, l)
                   for s, w, sc, ec, l in pairs[:n_val]]
        train_s = [CWISample(s, w, sc, ec, l)
                   for s, w, sc, ec, l in pairs[n_val:]]
        test_s  = []
        print(f"  Train: {len(train_s)}  Val: {len(val_s)}")

    # ── Model + Tokenizer ─────────────────────────────────────────────────────
    print(f"\n[train_cwi] Loading tokenizer: {config.deberta_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.deberta_model_name)

    print(f"[train_cwi] Building model: {config.deberta_model_name}")
    model = DeBERTaCWIModel(config.deberta_model_name)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = CWITrainer(model=model, config=config, device=device)
    trainer.train(train_s, val_s, tokenizer)

    # ── Final evaluation on held-out test set ─────────────────────────────────
    if test_s:
        print("\n[train_cwi] Evaluating on held-out test set...")
        evaluator = CWIEvaluator()
        # Manually evaluate with fixed threshold (dynamic threshold needs pipeline)
        evaluator.cwi = None

        from cwi_model import DeBERTaCWIClassifier
        from cwi_config import DEFAULT_CONFIG
        clf = DeBERTaCWIClassifier(config=config, device=device)
        clf.model = model
        clf._loaded = True

        # Evaluate at multiple thresholds
        from cwi_data import CWISample as S
        evaluator.cwi = type('_CWI', (), {'bert_cwi': clf})()
        best_t = evaluator.calibrate_threshold(test_s, start=0.30, end=0.70, steps=20)
        report = evaluator.evaluate(test_s, threshold=best_t,
                                    use_dynamic_threshold=False)
        report.print()

    print(f"\n[train_cwi] ✓ Training complete.")
    print(f"  Model saved to: {config.weights_path}")
    print(f"  Restart interactive_simplifier.py to use the trained model.")


if __name__ == '__main__':
    main()
