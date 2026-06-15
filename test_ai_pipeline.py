# test_ai_pipeline.py

import os
import torch
from ai_simplifier import AILexicalSimplifier, verify_bert_mlm


def run_ai_tests():
    config = {
        'bert_model': 'bert-base-uncased',
        'max_bert_tokens': 128
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing AILexicalSimplifier on device: {device}...")
    simplifier = AILexicalSimplifier(config, device)
    print("AILexicalSimplifier initialized successfully!\n")

    print("=" * 60)
    print("         BALANCED LEXICAL SIMPLIFICATION TESTS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # TEST 1 - Must simplify 'elegant'
    # ------------------------------------------------------------------
    tc1_input = "The nature is elegant today"
    print(f"\n--- TEST 1: '{tc1_input}' ---")
    print("Expected: elegant -> pretty / beautiful / lovely / nice\n")
    tc1_output = simplifier.simplify(tc1_input, verbose=True)

    # ------------------------------------------------------------------
    # TEST 2 - Simple sentence; should stay unchanged
    # ------------------------------------------------------------------
    tc2_input = "The cat sat on the mat"
    print(f"\n--- TEST 2: '{tc2_input}' ---")
    print("Expected: unchanged\n")
    tc2_output = simplifier.simplify(tc2_input, verbose=True)

    # ------------------------------------------------------------------
    # TEST 3 - physician -> doctor, comprehended -> understood
    # ------------------------------------------------------------------
    tc3_input = "The physician comprehended quickly"
    print(f"\n--- TEST 3: '{tc3_input}' ---")
    print("Expected: physician -> doctor, comprehended -> understood\n")
    tc3_output = simplifier.simplify(tc3_input, verbose=True)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("                   VERIFICATION REPORT")
    print("=" * 60)

    simple_synonyms = {"pretty", "beautiful", "lovely", "nice", "fine", "neat"}
    tc1_ok = any(syn in tc1_output.lower() for syn in simple_synonyms)
    tc2_ok = tc2_output.strip() == tc2_input.strip()
    tc3_ok = ("doctor" in tc3_output.lower() or "understood" in tc3_output.lower()
              or "physician" not in tc3_output.lower())

    def fmt(ok):
        return "PASSED [OK]" if ok else "FAILED [X]"

    print(f"Test 1 (elegant simplified):          {fmt(tc1_ok)}")
    print(f"  Before : {tc1_input}")
    print(f"  After  : {tc1_output}")

    print(f"\nTest 2 (simple sentence unchanged):   {fmt(tc2_ok)}")
    print(f"  Before : {tc2_input}")
    print(f"  After  : {tc2_output}")

    print(f"\nTest 3 (physician/comprehended):      {fmt(tc3_ok)}")
    print(f"  Before : {tc3_input}")
    print(f"  After  : {tc3_output}")

    # ------------------------------------------------------------------
    # Rule-free audit
    # ------------------------------------------------------------------
    print("\n--- Codebase Rule-Free Audit ---")
    modules_to_check = [
        "bert_surprisal.py", "bert_complexity.py", "dynamic_cwi.py",
        "bert_sense_disambiguator.py", "bert_candidate_generator.py",
        "bert_validator.py", "bert_ranker.py", "ai_simplifier.py"
    ]
    audit_passed = True
    for module in modules_to_check:
        if os.path.exists(module):
            with open(module, 'r', encoding='utf-8') as f:
                content = f.read()
            if "COMMON_WORDS" in content:
                print(f"  [FAIL] {module} references COMMON_WORDS!")
                audit_passed = False
            if "dale_chall" in content.lower():
                print(f"  [FAIL] {module} references dale_chall!")
                audit_passed = False

    if audit_passed:
        print("  [PASS] No hardcoded word-list references found.")

    all_ok = tc1_ok and tc2_ok and tc3_ok and audit_passed
    print("\n" + ("ALL TESTS PASSED!" if all_ok else "SOME TESTS FAILED. Check logs above."))


if __name__ == "__main__":
    run_ai_tests()
