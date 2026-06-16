# retrain_cwi.py
import os
import json
import random
import datasets  # Must be imported before torch/transformers to avoid Windows DLL crash
import torch
from transformers import AutoTokenizer

from cwi_config import CWIConfig
from cwi_data import CWISample
from cwi_model import DeBERTaCWIModel
from cwi_trainer import CWITrainer

def load_augmented_samples(path: str) -> list:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    samples = []
    for item in data:
        samples.append(CWISample(
            sentence=item["sentence"],
            word=item["word"],
            start_char=item["start_char"],
            end_char=item["end_char"],
            label=item["label"]
        ))
    return samples

def main():
    # Optimise CPU thread count for PyTorch
    torch.set_num_threads(4)

    config = CWIConfig(
        deberta_model_name      = "microsoft/deberta-v3-base",
        weights_path            = "cwi_bert_figurative_v2.pt",
        batch_size              = 16,
        learning_rate           = 2e-5,
        num_epochs              = 3,
        early_stopping_patience = 3,
        focal_gamma             = 2.0,
        focal_alpha             = 0.75,
        cwi_2018_train_dir      = "cwi2018_data",
        lex_mturk_path          = "lex_mturk.txt",
        benchls_path            = "BenchLS.txt",
        val_split               = 0.10,
        test_split              = 0.0,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[retrain_cwi] Device: {device}")

    # ── Load datasets ──────────────────────────────────────────────────────────
    print("[retrain_cwi] Loading LexMTurk and BenchLS datasets...")
    from data_loader import load_cwi_training_pairs
    base_pairs = load_cwi_training_pairs(config.lex_mturk_path, config.benchls_path)
    
    samples = [CWISample(s, w, sc, ec, l) for s, w, sc, ec, l in base_pairs]
    print(f"[retrain_cwi] Loaded {len(samples)} samples from LexMTurk + BenchLS.")

    print("[retrain_cwi] Loading augmented figurative/literal data...")
    if os.path.exists("augmented_cwi_data.json"):
        aug_samples = load_augmented_samples("augmented_cwi_data.json")
        samples.extend(aug_samples)
        print(f"[retrain_cwi] Added {len(aug_samples)} augmented samples. Total: {len(samples)}")
    else:
        print("[retrain_cwi] WARNING: augmented_cwi_data.json not found!")

    # Shuffle and split
    random.seed(42)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * config.val_split))
    val_s = samples[:n_val]
    train_s = samples[n_val:]

    print(f"[retrain_cwi] Split: {len(train_s)} train, {len(val_s)} val.")

    # ── Load model and starting weights ───────────────────────────────────────
    print(f"\n[retrain_cwi] Loading tokenizer: {config.deberta_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.deberta_model_name)

    print(f"[retrain_cwi] Building model: {config.deberta_model_name}")
    model = DeBERTaCWIModel(config.deberta_model_name)

    if os.path.exists("cwi_deberta.pt"):
        print("[retrain_cwi] Loading starting weights from cwi_deberta.pt...")
        model.load_state_dict(torch.load("cwi_deberta.pt", map_location=device))
    else:
        print("[retrain_cwi] WARNING: cwi_deberta.pt not found. Training from scratch.")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = CWITrainer(model=model, config=config, device=device)
    trainer.train(train_s, val_s, tokenizer)

    print(f"\n[retrain_cwi] [OK] Retraining complete. Model saved to {config.weights_path}")

if __name__ == "__main__":
    main()
