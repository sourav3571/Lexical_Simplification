========================================================================
             CONTEXT-AWARE HYBRID NEURAL LEXICAL SIMPLIFIER
========================================================================

This is a production-ready, 7-stage Lexical Simplification system utilizing
BERT contextual embeddings, Masked Language Modeling (MLM), WordNet
semantic gating, and dynamic contextual complexity metrics. This engine
dynamically simplifies complex words in English sentences without relying
on static dataset lookups.

Detailed pipeline diagram and justification of stages can be found in:
pipeline_architecture.txt

============================================================
                     WORKSPACE FILE LIST
============================================================

- config.py                  : Centralized thresholds and configurations
- preprocessing.py           : NLP preprocessing (spaCy content token extraction)
- bert_surprisal.py          : Context-sensitive surprisal calculations (MLM)
- bert_complexity.py         : Calibrated 3-feature context-aware complexity scorer
- dynamic_cwi.py             : Context-aware dynamic complex word identification
- bert_sense_disambiguator.py: Cosine-similarity based WordNet WSD using BERT
- bert_candidate_generator.py: Candidate generation & inherent complexity filtering
- bert_validator.py          : 4-stage validation gating for replacements
- bert_ranker.py             : PyTorch Gated Fusion Neural Ranker
- parallel_replacer.py       : Parallel word substitution utility
- ai_simplifier.py           : Main pipeline orchestration engine
- test_ai_pipeline.py        : Context-sensitive verification test suite
- interactive_simplifier.py  : Interactive CLI console for user input
- gated_fusion_ranker.pt     : Serialized neural ranker model weights
- BenchLS.txt                : Fallback human annotation dataset
- lex_mturk.txt              : Primary training annotation dataset
- requirements.txt           : Pipeline package dependencies
- pipeline_architecture.txt  : ASCII flowchart and design justification
- README.txt                 : This file

============================================================
                     GETTING STARTED
============================================================

1. Install package dependencies:
   pip install -r requirements.txt

2. Download WordNet and spaCy English model:
   python -c "import nltk; nltk.download('wordnet')"
   python -m spacy download en_core_web_sm

3. Run the context-sensitive verification test suite:
   python test_ai_pipeline.py

4. Run the interactive simplifies console:
   python interactive_simplifier.py
   (Follow the prompts to input any sentence you want to simplify, or enter '1' to exit)
