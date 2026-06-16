import datasets
import torch
import spacy
import wordfreq
from dynamic_cwi import DynamicContextualCWI
from cwi_config import DEFAULT_CONFIG

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None

    cwi = DynamicContextualCWI(nlp=nlp, device=device, config=DEFAULT_CONFIG)

    sentences = [
        "The boy went to school today.",
        "She poured spirit into the glass.",
        "The situation was increasingly precarious."
    ]

    for sentence in sentences:
        print(f"\nSentence: {sentence}")
        doc = nlp(sentence)
        content_tokens = [
            (t.text, t.pos_, t.idx, t.idx + len(t.text))
            for t in doc
            if t.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN')
            and t.text.isalpha()
        ]
        results = cwi.identify_complex_words(sentence, content_tokens)
        for r in results:
            print(f"  Word: {r['word']:15s} | Zipf: {r['word_zipf']:.2f} | BERT: {r['bert_score']:.3f} | Drift: {r['drift_score']:.3f} | Pattern: {r['pattern_detected']:10s} | Struct: {r['structural_score']:.1f} | Boost: {r['pattern_boost']:.2f} | Score: {r['ensemble_score']:.3f} | Thresh: {r['effective_threshold']:.3f} | Complex: {r['is_complex']}")

if __name__ == "__main__":
    main()
