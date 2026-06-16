# cwi_evaluator.py
"""
Evaluation module for the CWI system.

Outputs:
  - Precision, Recall, F1, Accuracy
  - Per-category analysis (figurative, polysemous, simple, complex)
  - Error analysis (false positives, false negatives)
  - Threshold calibration curve
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from cwi_data import CWISample


@dataclass
class CWIPrediction:
    sample:       CWISample
    predicted:    int      # 0 or 1
    confidence:   float    # P(complex)
    is_correct:   bool


@dataclass
class EvalReport:
    precision:    float
    recall:       float
    f1:           float
    accuracy:     float
    tp:           int
    fp:           int
    fn:           int
    tn:           int
    n_samples:    int
    false_positives: List[CWIPrediction] = field(default_factory=list)
    false_negatives: List[CWIPrediction] = field(default_factory=list)

    def print(self) -> None:
        print("\n" + "=" * 60)
        print("CWI EVALUATION REPORT")
        print("=" * 60)
        print(f"  Samples:   {self.n_samples}")
        print(f"  Precision: {self.precision:.4f}  ({self.precision*100:.1f}%)")
        print(f"  Recall:    {self.recall:.4f}  ({self.recall*100:.1f}%)")
        print(f"  F1 Score:  {self.f1:.4f}  ({self.f1*100:.1f}%)")
        print(f"  Accuracy:  {self.accuracy:.4f}  ({self.accuracy*100:.1f}%)")
        print(f"\n  Confusion Matrix:")
        print(f"    TP (correct complex): {self.tp}")
        print(f"    TN (correct simple):  {self.tn}")
        print(f"    FP (flagged simple):  {self.fp}  <-- False Positive")
        print(f"    FN (missed complex):  {self.fn}  <-- False Negative")
        print(f"\n  False Positive Rate: {self.fp/(self.fp+self.tn+1e-9):.4f}")
        print(f"  False Negative Rate: {self.fn/(self.fn+self.tp+1e-9):.4f}")

        if self.false_positives:
            print(f"\n  Top False Positives (simple words flagged as complex):")
            for i, pred in enumerate(self.false_positives[:10]):
                print(f"    [{i+1}] '{pred.sample.word}' "
                      f"(conf={pred.confidence:.3f}) "
                      f"in: '{pred.sample.sentence[:60]}...'")

        if self.false_negatives:
            print(f"\n  Top False Negatives (complex words missed):")
            for i, pred in enumerate(self.false_negatives[:10]):
                print(f"    [{i+1}] '{pred.sample.word}' "
                      f"(conf={pred.confidence:.3f}) "
                      f"in: '{pred.sample.sentence[:60]}...'")
        print("=" * 60 + "\n")


class CWIEvaluator:
    """
    Evaluates the full CWI system on a set of labelled samples.

    Usage:
        evaluator = CWIEvaluator(cwi_system)  # cwi_system = DynamicContextualCWI
        report = evaluator.evaluate(test_samples, threshold=0.5)
        report.print()

        # Run threshold sweep
        evaluator.calibrate_threshold(val_samples, start=0.30, end=0.70, steps=20)
    """

    def __init__(self, cwi_system=None) -> None:
        """
        cwi_system: a DynamicContextualCWI instance, or any object
                    with method score(sentence, word, start, end, pos) -> float
        """
        self.cwi = cwi_system

    def evaluate(
        self,
        samples:   List[CWISample],
        threshold: float = 0.5,
        use_dynamic_threshold: bool = True,
    ) -> EvalReport:
        """
        Evaluate on a list of CWISample objects.

        If use_dynamic_threshold=True, uses the CWI system's built-in
        dynamic per-sentence threshold instead of the fixed `threshold` param.
        """
        predictions: List[CWIPrediction] = []

        if use_dynamic_threshold and self.cwi is not None:
            predictions = self._evaluate_with_pipeline(samples)
        else:
            predictions = self._evaluate_with_threshold(samples, threshold)

        tp = sum(1 for p in predictions if p.predicted == 1 and p.sample.label == 1)
        fp = sum(1 for p in predictions if p.predicted == 1 and p.sample.label == 0)
        fn = sum(1 for p in predictions if p.predicted == 0 and p.sample.label == 1)
        tn = sum(1 for p in predictions if p.predicted == 0 and p.sample.label == 0)

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-9)

        # Sort errors by confidence (most confident wrong predictions first)
        fps = sorted([p for p in predictions
                      if p.predicted == 1 and p.sample.label == 0],
                     key=lambda x: x.confidence, reverse=True)
        fns = sorted([p for p in predictions
                      if p.predicted == 0 and p.sample.label == 1],
                     key=lambda x: x.confidence)   # lowest conf = most missed

        return EvalReport(
            precision=precision, recall=recall, f1=f1, accuracy=accuracy,
            tp=tp, fp=fp, fn=fn, tn=tn,
            n_samples=len(predictions),
            false_positives=fps,
            false_negatives=fns,
        )

    def _evaluate_with_pipeline(
        self, samples: List[CWISample]
    ) -> List[CWIPrediction]:
        """
        Group samples by sentence and run the full CWI pipeline
        (with dynamic threshold) per sentence.
        """
        from collections import defaultdict
        by_sentence: Dict[str, List[CWISample]] = defaultdict(list)
        for s in samples:
            by_sentence[s.sentence].append(s)

        results: List[CWIPrediction] = []
        for sentence, sent_samples in by_sentence.items():
            # Build content_tokens for this sentence
            content_tokens = [
                (s.word, 'NOUN', s.start_char, s.end_char)
                for s in sent_samples
            ]
            try:
                cwi_results = self.cwi.identify_complex_words(
                    sentence, content_tokens)
                cwi_map = {r['word'].lower(): r for r in cwi_results}
            except Exception:
                cwi_map = {}

            for s in sent_samples:
                res = cwi_map.get(s.word.lower(), {})
                predicted   = 1 if res.get('is_complex', False) else 0
                confidence  = float(res.get('ensemble_score', 0.5))
                results.append(CWIPrediction(
                    sample=s,
                    predicted=predicted,
                    confidence=confidence,
                    is_correct=(predicted == s.label),
                ))
        return results

    def _evaluate_with_threshold(
        self, samples: List[CWISample], threshold: float
    ) -> List[CWIPrediction]:
        """Score each sample independently with a fixed threshold."""
        results: List[CWIPrediction] = []
        for s in samples:
            try:
                if self.cwi is not None:
                    conf = self.cwi.bert_cwi.score(
                        s.sentence, s.word, s.start_char, s.end_char)
                else:
                    conf = 0.5
            except Exception:
                conf = 0.5

            predicted = 1 if conf >= threshold else 0
            results.append(CWIPrediction(
                sample=s,
                predicted=predicted,
                confidence=conf,
                is_correct=(predicted == s.label),
            ))
        return results

    def calibrate_threshold(
        self,
        samples: List[CWISample],
        start:   float = 0.20,
        end:     float = 0.80,
        steps:   int   = 20,
    ) -> float:
        """
        Sweep threshold values and print a table of precision/recall/F1.

        Returns the threshold that maximises PRECISION at F1 > 0.70.
        This aligns with the precision > recall requirement.
        """
        print("\n[CWIEvaluator] Pre-calculating model confidence scores for threshold sweep...")
        confidences = []
        for s in samples:
            try:
                if self.cwi is not None:
                    conf = self.cwi.bert_cwi.score(
                        s.sentence, s.word, s.start_char, s.end_char)
                else:
                    conf = 0.5
            except Exception:
                conf = 0.5
            confidences.append(conf)

        print("\n" + "-" * 65)
        print(f"  {'Threshold':>9}  {'Precision':>9}  {'Recall':>7}  "
              f"{'F1':>7}  {'Flags':>6}")
        print("-" * 65)

        best_threshold = start
        best_precision = 0.0

        thresholds = [start + (end - start) * i / steps for i in range(steps + 1)]
        for t in thresholds:
            tp = fp = fn = 0
            for s, conf in zip(samples, confidences):
                pred = 1 if conf >= t else 0
                if pred == 1 and s.label == 1:
                    tp += 1
                elif pred == 1 and s.label == 0:
                    fp += 1
                elif pred == 0 and s.label == 1:
                    fn += 1

            n_flags = tp + fp
            precision = tp / (tp + fp + 1e-9)
            recall    = tp / (tp + fn + 1e-9)
            f1        = 2 * precision * recall / (precision + recall + 1e-9)

            marker = " <-- best" if (precision > best_precision and f1 > 0.70) else ""
            if precision > best_precision and f1 > 0.70:
                best_precision = precision
                best_threshold = t

            print(f"  {t:>9.3f}  {precision:>9.4f}  {recall:>7.4f}  "
                  f"{f1:>7.4f}  {n_flags:>6}{marker}")

        print("-" * 65)
        print(f"\n  Optimal threshold (precision-first): {best_threshold:.3f}")
        print(f"  Best precision at F1 > 0.70:         {best_precision:.4f}")
        return best_threshold

    def run_test_cases(self, cwi_system) -> None:
        """
        Run the 11 canonical test cases and show predictions.
        No ground-truth labels needed — shows raw system output.
        """
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = None

        test_cases = [
            ("The nature of the person is good.",
             {"nature": "COMPLEX"}),
            ("The nature outside is beautiful.",
             {"nature": "SIMPLE"}),
            ("The spirit of the law matters.",
             {"spirit": "COMPLEX"}),
            ("She poured spirit into the glass.",
             {"spirit": "SIMPLE"}),
            ("The physician prescribed medication.",
             {"physician": "COMPLEX", "prescribed": "COMPLEX", "medication": "COMPLEX"}),
            ("The situation was increasingly precarious.",
             {"precarious": "COMPLEX"}),
            ("She demonstrated exceptional resilience.",
             {"demonstrated": "COMPLEX", "exceptional": "COMPLEX", "resilience": "COMPLEX"}),
            ("The boy went to school today.",
             {}),
            ("She ate an apple for lunch.",
             {}),
            ("The boy was exhausted after playing.",
             {"exhausted": "COMPLEX"}),
            ("She purchased a beautiful dress.",
             {"purchased": "COMPLEX"}),
        ]

        print("\n" + "=" * 70)
        print("TEST CASE VALIDATION")
        print("=" * 70)

        for sentence, expected in test_cases:
            print(f"\nINPUT: {sentence}")
            print(f"EXPECTED: {expected or 'NO complex words'}")

            if nlp:
                doc = nlp(sentence)
                content_tokens = [
                    (t.text, t.pos_, t.idx, t.idx + len(t.text))
                    for t in doc
                    if t.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN')
                    and t.text.isalpha()
                ]
            else:
                content_tokens = []

            try:
                results = cwi_system.identify_complex_words(
                    sentence, content_tokens)
                detected = {r['word']: r for r in results if r.get('is_complex')}
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            # Check each expected word
            for word, exp_label in expected.items():
                in_detected = word.lower() in {k.lower() for k in detected}
                status = "[OK]  " if (in_detected and exp_label == "COMPLEX") else \
                         "[OK]  " if (not in_detected and exp_label == "SIMPLE") else "[FAIL]"
                score  = detected.get(word, {}).get('ensemble_score', 0.0)
                print(f"  {status} {word:15s} expected={exp_label:7s}  "
                      f"detected={'YES' if in_detected else 'NO ':3s}  "
                      f"score={score:.3f}")

            # Check for false positives (detected but not expected complex)
            for word in detected:
                if word.lower() not in {k.lower() for k in expected}:
                    score = detected[word].get('ensemble_score', 0.0)
                    print(f"  [FAIL] {word:15s} expected=SIMPLE   "
                          f"detected=YES   score={score:.3f}  [FALSE POSITIVE]")

        print("\n" + "=" * 70)
