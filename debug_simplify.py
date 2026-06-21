import sys
import os
import torch
from ai_simplifier import AILexicalSimplifier

def main():
    device = torch.device("cpu")
    config = {
        'bert_model': 'bert-base-uncased',
        'max_bert_tokens': 128,
        'FIG_CONFIG': {
            "idiom_confidence_threshold": 0.80,
            "use_database_lookup": True,
            "use_bert_classifier": True,
            "database_priority": True,
            "metaphor_threshold": 0.60,
            "roberta_weight": 0.60,
            "sbert_drift_weight": 0.40,
            "structural_boost": 1.30,
            "drift_override": 0.38,
            "min_semantic_sim": 0.75,
            "min_zipf_gain": 0.5,
            "prefer_concrete": True,
            "idiom_first": True,
            "metaphor_second": True,
            "standard_cwi_third": True
        }
    }
    
    simplifier = AILexicalSimplifier(config, device=device)
    
    test_sentences = [
        "The boy was exhausted.",
        "She purchased a dress.",
        "He obtained permission.",
        "The girl was delighted.",
        "The situation was dire.",
        "The task was arduous.",
        "The methodology was comprehensive.",
        "The surgical procedure was successful."
    ]
    
    for sent in test_sentences:
        print("\n" + "="*80)
        print(f"INPUT SENTENCE: {sent}")
        print("="*80)
        output = simplifier.simplify(sent, verbose=True)
        print(f"FINAL OUTPUT:   {output}")

if __name__ == "__main__":
    main()
