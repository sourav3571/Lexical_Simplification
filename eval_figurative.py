# eval_figurative.py

import os
import time
import torch
from data_prep import prepare_all_data
from idiom_classifier import train_idiom_classifier
from metaphor_detector import train_metaphor_detector
from ai_simplifier import AILexicalSimplifier

# Colour codes for console output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def train_models_if_needed():
    """Trains quick versions of the models if weights do not exist yet."""
    # Ensure data splits exist
    if not os.path.exists("data/idiom_train.json") or not os.path.exists("data/metaphor_train.json"):
        prepare_all_data()

    if not os.path.exists("idiom_classifier.pt"):
        print("\n--- Training Idiom Classifier (5 epochs, max_length=32) ---")
        train_idiom_classifier(epochs=5, batch_size=8, max_train_samples=120)
    else:
        print("\n--- Idiom Classifier weights found. Skipping training. ---")

    if not os.path.exists("metaphor_detector.pt"):
        print("\n--- Training Metaphor Detector (5 epochs, max_length=32) ---")
        train_metaphor_detector(epochs=5, batch_size=8, max_train_samples=120)
    else:
        print("\n--- Metaphor Detector weights found. Skipping training. ---")

def run_evaluation():
    train_models_if_needed()

    print("\n--- Initializing Lexical Simplifier Pipeline ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Config
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
    print("Lexical Simplifier successfully loaded.\n")

    # Defined target test cases
    test_cases = [
        # (category, input_sentence, expected_output, should_change)
        
        # 1. Idiom detection & simplification
        ("Idiom", "He kicked the bucket last year.", "He died last year.", True),
        ("Idiom", "She is under the weather today.", "She is feeling sick today.", True),
        ("Idiom", "He spilled the beans about the plan.", "He revealed the secret about the plan.", True),
        
        # 2. Metaphor detection & resolution
        ("Metaphor", "The face of poverty is visible.", "The reality of poverty is visible.", True),
        ("Metaphor", "The heart of the problem is trust.", "The core of the problem is trust.", True),
        
        # 3. Figurative Adjectives
        ("Adjective", "The strength is enduring.", "The strength is lasting.", True),
        ("Adjective", "The pain was excruciating.", "The pain was severe.", True),
        
        # 4. Same Word Context (Literal/No Change)
        ("Literal (Same Word)", "The nature outside is beautiful.", "The nature outside is beautiful.", False),
        ("Literal (Same Word)", "The bank near the river flooded.", "The bank near the river flooded.", False),
        ("Literal (Same Word)", "His heart beats very fast.", "His heart beats very fast.", False),
        ("Literal (Same Word)", "The face was very beautiful.", "The face was very beautiful.", False),
        
        # 5. Standard Regression Protection
        ("Regression Protection", "The boy went to school.", "The boy went to school.", False),
        ("Regression Protection", "The physician treated the patient.", "The doctor treated the patient.", True),
        ("Regression Protection", "She utilized the equipment.", "She used the equipment.", True),
    ]

    print(f"{BOLD}{'Category':<25} | {'Input Sentence':<40} | {'Expected Output':<40} | {'Status'}{RESET}")
    print("-" * 125)

    tp, fp, tn, fn = 0, 0, 0, 0
    results_table = []

    for category, input_sent, expected, should_change in test_cases:
        # Run through simplifier
        output = simplifier.simplify(input_sent, verbose=False)
        
        # Clean spacing for comparison
        clean_output = " ".join(output.strip().split())
        clean_expected = " ".join(expected.strip().split())
        
        is_correct = (clean_output == clean_expected)
        
        # Metrics logic:
        # True Positive (TP): should change, changed correctly
        # False Positive (FP): should NOT change, but changed OR changed incorrectly
        # True Negative (TN): should NOT change, stayed same (correct)
        # False Negative (FN): should change, but stayed same OR changed incorrectly
        
        if should_change:
            if is_correct:
                tp += 1
                status = f"{GREEN}PASS (Correct Change){RESET}"
            else:
                fn += 1
                status = f"{RED}FAIL (Incorrect/No Change){RESET}"
        else:
            if is_correct:
                tn += 1
                status = f"{GREEN}PASS (Correctly Kept){RESET}"
            else:
                fp += 1
                status = f"{RED}FAIL (Incorrectly Changed){RESET}"

        results_table.append({
            "category": category,
            "input": input_sent,
            "output": output,
            "expected": expected,
            "status": status
        })
        
        print(f"{category:<25} | {input_sent:<40} | {expected:<40} | {status}")

    # Calculations
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 60)
    print(f"{BOLD}EVALUATION METRICS SUMMARY:{RESET}")
    print("=" * 60)
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Negatives (FN): {fn}")
    print("-" * 30)
    print(f"Precision:            {precision:.4f} ({precision * 100:.1f}%)")
    print(f"Recall:               {recall:.4f} ({recall * 100:.1f}%)")
    print(f"F1 Score:             {f1:.4f} ({f1 * 100:.1f}%)")
    print("=" * 60)

    # Save details to markdown artifact
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("# Figurative Language Simplifier Evaluation Report\n\n")
        f.write("## Performance Metrics\n\n")
        f.write("| Metric | Value | Detail |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| Precision | {precision*100:.2f}% | {tp}/{tp+fp} correct simplifications |\n")
        f.write(f"| Recall | {recall*100:.2f}% | {tp}/{tp+fn} target figurative cases detected |\n")
        f.write(f"| F1 Score | {f1*100:.2f}% | Harmonic mean of Precision and Recall |\n\n")
        
        f.write("## Confusion Matrix Elements\n\n")
        f.write(f"- **True Positives (TP)**: {tp} (Correct simplifications of idioms/metaphors/complex words)\n")
        f.write(f"- **False Positives (FP)**: {fp} (Literal/simple contexts mistakenly simplified)\n")
        f.write(f"- **True Negatives (TN)**: {tn} (Literal/simple contexts correctly preserved)\n")
        f.write(f"- **False Negatives (FN)**: {fn} (Target cases missed or simplified incorrectly)\n\n")
        
        f.write("## Test Cases Comparison Table\n\n")
        f.write("| Category | Input Sentence | Expected Output | Actual Output | Status |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for res in results_table:
            # Strip colour tags for markdown
            clean_status = res["status"].replace(GREEN, "").replace(RED, "").replace(RESET, "").replace(BOLD, "")
            f.write(f"| {res['category']} | {res['input']} | {res['expected']} | {res['output']} | {clean_status} |\n")

    print(f"\nSaved detailed evaluation report artifact to: {os.path.abspath('artifacts/evaluation_report.md')}")

if __name__ == "__main__":
    run_evaluation()
