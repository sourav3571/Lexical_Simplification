# test_cwi_model.py
import datasets
import torch
import spacy
from dynamic_cwi import DynamicContextualCWI
from cwi_evaluator import CWIEvaluator
from cwi_config import DEFAULT_CONFIG

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None

    cwi_system = DynamicContextualCWI(nlp=nlp, device=device, config=DEFAULT_CONFIG)
    
    print("\n--- CONFIG DETAILS ---")
    print(f"short_sentence_threshold: {cwi_system.cfg.short_sentence_threshold}")
    print(f"zipf_always_simple: {cwi_system.cfg.zipf_always_simple}")
    print(f"dynamic_k_standard: {cwi_system.cfg.dynamic_k_standard}")
    print("----------------------\n")

    print("\nStarting Test Case Validation...")
    evaluator = CWIEvaluator()
    evaluator.run_test_cases(cwi_system)

if __name__ == "__main__":
    main()
