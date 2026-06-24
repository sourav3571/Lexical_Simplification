# test_integration.py
import os
import sys
import torch

def test_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'bert_model': 'bert-base-uncased',
        'max_bert_tokens': 128,
        'lex_mturk_path': 'lex_mturk.txt',
        'benchls_path': 'BenchLS.txt',
        'ppdb_path': 'ppdb_fallback.json',
    }

    print("--- Initializing AILexicalSimplifier ---")
    from ai_simplifier import AILexicalSimplifier
    simplifier = AILexicalSimplifier(config, device)
    print("[OK] Initialized successfully")

    # We will test an idiom and a complex word
    test_sentence = "He was disappointed because he kicked the bucket yesterday."
    print(f"\n--- Testing Simplification on: '{test_sentence}' ---")
    simplified = simplifier.simplify(test_sentence, verbose=True)
    print(f"Result: '{simplified}'")

    print("\n--- Visual Metadata generated: ---")
    import json
    metadata = simplifier.last_visual_data
    if metadata:
        try:
            print(json.dumps(metadata, indent=2))
        except Exception as e:
            print("Could not print metadata json directly due to encoding, but metadata exists.")
        print("[OK] Visual metadata exists!")
    else:
        print("[ERROR] Visual metadata is None!")

if __name__ == "__main__":
    test_pipeline()
