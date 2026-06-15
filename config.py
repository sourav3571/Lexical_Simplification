import os
from typing import Dict, Any

# Global configuration dictionary for hyperparameters, model paths, and resource configurations
CONFIG: Dict[str, Any] = {
    'bert_model': 'bert-base-uncased',
    'glove_model': 'glove-wiki-gigaword-100',
    'batch_size': 16,
    'epochs': 10,
    'lr_bert': 2e-5,
    'lr_ranker': 1e-3,
    'max_length': 32,
    'dropout': 0.3,
    'freq_threshold': 4.0,
    'simp_threshold': 0.35,  # Combined complexity threshold for CWI (Stage 2)
    'glove_dim': 100,
    'seed': 42,
    'val_split': 0.1,
    'grad_clip': 1.0,
    'weight_decay': 0.01,
    'fine_tune_bert': False,
    'dale_chall_path': 'dale_chall.txt',
    'oxford3000_path': 'oxford3000.txt',
    'lex_mturk_path': 'lex_mturk.txt',
    'best_model_path': 'best_model.pt',
    'max_bert_tokens': 128
}
