========================================================================
            CONTEXT-AWARE HYBRID NEURAL LEXICAL SIMPLIFIER
========================================================================

This is a production-ready, 6-stage Lexical Simplification system utilizing
BERT contextual embeddings, Masked Language Modeling (MLM), WordNet
semantic gating, and GloVe vector spaces. This engine dynamically simplifies
complex words in English sentences without relying on static dataset lookups.

Detailed pipeline diagram and justification of stages can be found in:
pipeline_architecture.txt

============================================================
                     WORKSPACE FILE LIST
============================================================

- config.py                     : Centralized thresholds and configurations
- preprocessing.py              : NLP preprocessing (spaCy)
- contextual_cwi.py             : Context-aware Complex Word Identification
- word_sense_disambiguation.py  : WordNet-based WSD using BERT embeddings
- candidate_generator.py        : Candidate generation & semantic gating
- model.py                      : PyTorch Neural Ranker network definition
- dataset.py                    : PyTorch dataset parser for BenchLS
- inference.py                  : Interactive command-line inference engine
- train.py                      : Training script for modular setup
- test.py                       : Master verification test suite (8/8 tests)
- best_model.pt                 : Trained PyTorch neural ranking weights
- BenchLS.txt                   : BenchLS lexical simplification dataset
- dale_chall.txt                : Dale-Chall easy word list
- oxford3000.txt                : Oxford 3000 familiar word list
- requirements.txt              : Pipeline package dependencies
- pipeline_architecture.txt     : ASCII flowchart and design justification
- README.txt                    : This file


============================================================
                     GETTING STARTED
============================================================

1. Install package dependencies:
   pip install -r requirements.txt

2. Download WordNet and spaCy English model:
   python -c "import nltk; nltk.download('wordnet')"
   python -m spacy download en_core_web_sm

3. Run the 8-stage verification test suite:
   python test.py

4. Run the interactive simplifies console:
   python inference.py
   (Follow the prompts to input any sentence you want to simplify)
