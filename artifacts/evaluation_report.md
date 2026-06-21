# Figurative Language Simplifier Evaluation Report

## Performance Metrics

| Metric | Value | Detail |
| --- | --- | --- |
| Precision | 100.00% | 9/9 correct simplifications |
| Recall | 100.00% | 9/9 target figurative cases detected |
| F1 Score | 100.00% | Harmonic mean of Precision and Recall |

## Confusion Matrix Elements

- **True Positives (TP)**: 9 (Correct simplifications of idioms/metaphors/complex words)
- **False Positives (FP)**: 0 (Literal/simple contexts mistakenly simplified)
- **True Negatives (TN)**: 5 (Literal/simple contexts correctly preserved)
- **False Negatives (FN)**: 0 (Target cases missed or simplified incorrectly)

## Test Cases Comparison Table

| Category | Input Sentence | Expected Output | Actual Output | Status |
| --- | --- | --- | --- | --- |
| Idiom | He kicked the bucket last year. | He died last year. | He died last year. | PASS (Correct Change) |
| Idiom | She is under the weather today. | She is feeling sick today. | She is feeling sick today. | PASS (Correct Change) |
| Idiom | He spilled the beans about the plan. | He revealed the secret about the plan. | He revealed the secret about the plan. | PASS (Correct Change) |
| Metaphor | The face of poverty is visible. | The reality of poverty is visible. | The reality of poverty is visible. | PASS (Correct Change) |
| Metaphor | The heart of the problem is trust. | The core of the problem is trust. | The core of the problem is trust. | PASS (Correct Change) |
| Adjective | The strength is enduring. | The strength is lasting. | The strength is lasting. | PASS (Correct Change) |
| Adjective | The pain was excruciating. | The pain was severe. | The pain was severe. | PASS (Correct Change) |
| Literal (Same Word) | The nature outside is beautiful. | The nature outside is beautiful. | The nature outside is beautiful. | PASS (Correctly Kept) |
| Literal (Same Word) | The bank near the river flooded. | The bank near the river flooded. | The bank near the river flooded. | PASS (Correctly Kept) |
| Literal (Same Word) | His heart beats very fast. | His heart beats very fast. | His heart beats very fast. | PASS (Correctly Kept) |
| Literal (Same Word) | The face was very beautiful. | The face was very beautiful. | The face was very beautiful. | PASS (Correctly Kept) |
| Regression Protection | The boy went to school. | The boy went to school. | The boy went to school. | PASS (Correctly Kept) |
| Regression Protection | The physician treated the patient. | The doctor treated the patient. | The doctor treated the patient. | PASS (Correct Change) |
| Regression Protection | She utilized the equipment. | She used the equipment. | She used the equipment. | PASS (Correct Change) |
