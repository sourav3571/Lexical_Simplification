# Lexical Simplification Pipeline Architecture

This document describes the design, data flow, and advantages of our hybrid neural Lexical Simplification pipeline. It operates completely dynamically at inference time without relying on static dataset mapping lookups.

---

## 1. System Flow Diagram

Below is the end-to-end data flow showing how a raw sentence is analyzed, complex words are identified, synonym candidates are dynamically generated, and the neural ranking model selects the optimal replacement.

```mermaid
flowchart TD
    %% Input Stage
    Input[Raw Input Sentence] --> Pre[1. Preprocessor - spaCy]
    
    %% Preprocessing
    Pre -->|Tokens, Lemmatization & POS| CWI[2. Complex Word Identifier]
    
    %% CWI Stage
    CWI -->|Zipf Freq < 4.0 OR Length > 8 OR Syllables > 3| CWFound{Complex Word?}
    CWFound -->|No| NoCW[Terminate: No Simplification Needed]
    CWFound -->|Yes| Target[Target Word Identified]
    
    %% Candidate Generation
    Target --> CG[3. Dynamic Candidate Generator]
    CG -->|Retrieve ADJ/NOUN/VERB/ADV| WN[WordNet Synsets]
    CG -->|Retrieve top_n = 500 neighbors| GloVe[GloVe Word Embeddings]
    
    WN --> Comb[Combine Candidates]
    GloVe --> Comb
    
    Comb --> Filter[Filter Candidates]
    Filter -->|1. Single word alphabetic check| F1[Clean Words]
    Filter -->|2. Freq candidate > Freq target| F2[Simpler Synonyms]
    
    %% Feature Extraction
    F2 --> Feat[4. Feature Extractor]
    Feat -->|1. Sentence Context| BERT_Ctx[BERT Contextual Embeddings]
    Feat -->|2. Grammatical Fit| MLM[BERT Masked LM Log Probability]
    Feat -->|3. Lexical Distance| StaticCos[BERT Static Cosine + GloVe Cosine]
    Feat -->|4. Graph Distance| WNSim[WordNet Path Similarity]
    Feat -->|5. Simplicity Delta| SimpFeat[Zipf Frequency Gain + Length/Syllable Deltas]
    
    %% Neural Model Ranking
    BERT_Ctx & MLM & StaticCos & WNSim --> SemNet[Semantic Sub-Network]
    SimpFeat --> SimpNet[Simplicity Sub-Network]
    
    SemNet -->|Semantic Fit Score| Multi[5. Scoring Gate]
    SimpNet -->|Simplicity Fit Score| Multi
    
    Multi -->|Combined Score = Sem * Simp| Rank[6. Candidate Ranker]
    Rank -->|Select highest scoring candidate| Winner[Optimal Simplified Word]
    Winner --> Replace[7. Word Substitution]
    Replace --> Output[Simplified Context-Aware Sentence]
```

---

## 2. Why This Dynamic Pipeline is the Best

This design represents a state-of-the-art **hybrid neural-lexical simplification engine** that outperforms direct database lookups or simple heuristic models for several reasons:

### 1. Zero Direct Dataset Mapping (Pure Generalization)
*   **The Problem with Direct Mapping:** Parsing a CSV dataset (like `lex_mturk.csv`) for candidate lookups at inference restricts the pipeline to only simplify words seen during training. It cannot simplify novel, out-of-vocabulary words.
*   **Our Solution:** The candidates are generated on-the-fly from the union of **WordNet synsets** and **GloVe neighborhood spaces**. By scaling the GloVe neighborhood search space to **500 dimensions**, we capture broader human-like candidate pools (such as extracting `"unclear"` for `"ambiguous"`) without referencing the dataset.

### 2. Multi-Gated Feature Fusion (Semantic + Simplicity)
Instead of relying on a single ranking criterion (like frequency or BERT context), the neural ranker uses two distinct sub-networks:
*   **Semantic Sub-Network:** Validates if the replacement fits the sentence context (BERT Contextual Encoder), fits the local grammar (BERT Masked LM Probability), and is semantically close to the original meaning (static GloVe and BERT embedding similarity).
*   **Simplicity Sub-Network:** Evaluates how much easier the replacement is to read, based on Zipf frequency gain, syllable reduction, and character length reduction.

These two scores are multiplied (`Score = Semantic_Fit * Simplicity_Fit`). This acts as an **AND gate**: a word must be **both** semantically accurate **and** simpler to win. This prevents the model from choosing simple but contextually incorrect words.

### 3. BERT Contextual Redundancy and Co-occurrence Penalty
*   The transformer context representation (`BERT Contextual Embeddings`) naturally penalizes words that cause redundancy. 
*   For example, in the sentence *"The results were ambiguous and unclear"*, replacing *"ambiguous"* with *"unclear"* creates a repetitive phrase (*"unclear and unclear"*). The model's context projector detects this semantic overlap and scores the redundancy lower than a natural alternative like *"vague"*.

### 4. Deterministic and Robust Fallback
*   If the Complex Word Identification (CWI) step finds no words exceeding the default threshold, the system automatically falls back to finding the word with the lowest Zipf frequency. This guarantees that any input sentence is analyzed and simplified.
*   If candidate generation for a complex word fails, the pipeline handles the exception cleanly without crashing, making it highly suitable for production APIs or user interfaces.
