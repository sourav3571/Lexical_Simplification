# data_loader.py
"""
Loads LexMTurk (lex_mturk.txt) and BenchLS (BenchLS.txt) datasets.

Output of build_gold_table():
    {
        "physician": ["doctor", "surgeon", "medic"],
        "nutritious": ["healthy", "nourishing", "wholesome"],
        ...
    }

Output of load_cwi_training_pairs():
    List of (sentence, word, start_char, end_char, label)
    where label = 1 (complex) / 0 (simple).
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_line(line: str) -> Tuple[str, str, List[Tuple[str, int]]]:
    """
    Parse one tab-separated line from either dataset.

    LexMTurk format:
        sentence TAB target TAB count TAB cand1:votes TAB cand2:votes ...
    BenchLS format:
        sentence TAB target TAB count TAB cand1 TAB cand2 ...   (vote = 1 each)

    Returns: (sentence, target, [(candidate, votes), ...])
    """
    parts = line.strip().split('\t')
    if len(parts) < 3:
        return '', '', []

    sentence = parts[0].strip().strip('"')
    target   = parts[1].strip().lower()

    # Determine if columns contain vote counts ("word:N")
    raw_cands = parts[3:] if (len(parts) > 3 and parts[2].isdigit()) else parts[2:]
    has_colons = any(':' in p for p in raw_cands)

    candidates: List[Tuple[str, int]] = []
    if has_colons:
        for item in raw_cands:
            if ':' in item:
                tok = item.split(':')
                word  = tok[0].strip().lower()
                try:
                    votes = int(tok[1])
                except (IndexError, ValueError):
                    votes = 1
                if word.isalpha():
                    candidates.append((word, votes))
    else:
        from collections import Counter
        counts = Counter(p.strip().lower() for p in raw_cands if p.strip().isalpha())
        candidates = list(counts.items())

    return sentence, target, candidates


def _load_file(path: str) -> List[Tuple[str, str, List[Tuple[str, int]]]]:
    if not os.path.exists(path):
        print(f"[data_loader] WARNING: file not found: {path}")
        return []
    rows = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip():
                continue
            sent, target, cands = _parse_line(line)
            if sent and target and cands:
                rows.append((sent, target, cands))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_gold_table(
    lex_mturk_path: str = 'lex_mturk.txt',
    benchls_path:   str = 'BenchLS.txt',
    min_votes: int = 2
) -> Dict[str, List[str]]:
    """
    Build a gold lookup table:  {complex_word: [ranked_simple_substitutions]}

    Candidates are ranked by total vote count across both datasets.
    Only candidates with >= min_votes are included (filters noise).

    Usage:
        gold = build_gold_table()
        subs = gold.get("physician", [])   # → ["doctor", "surgeon", ...]
    """
    vote_table: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for path in [lex_mturk_path, benchls_path]:
        for _, target, cands in _load_file(path):
            for cand, votes in cands:
                if cand != target:
                    vote_table[target][cand] += votes

    gold: Dict[str, List[str]] = {}
    for word, cand_votes in vote_table.items():
        ranked = sorted(
            ((c, v) for c, v in cand_votes.items() if v >= min_votes),
            key=lambda x: x[1],
            reverse=True
        )
        if ranked:
            gold[word] = [c for c, _ in ranked]

    print(f"[data_loader] Gold table built: {len(gold)} complex words, "
          f"{sum(len(v) for v in gold.values())} total substitutions.")
    return gold


def load_cwi_training_pairs(
    lex_mturk_path: str = 'lex_mturk.txt',
    benchls_path:   str = 'BenchLS.txt'
) -> List[Tuple[str, str, int, int, int]]:
    """
    Build binary CWI training pairs from both datasets.

    Strategy:
      - Target word = COMPLEX (label 1).  These are words humans chose to
        simplify — i.e., they are definitionally non-trivial.
      - Top-1 substitute = SIMPLE (label 0).  The winning human substitute
        is definitionally simpler and serves as a negative example.

    Returns: list of (sentence, word, start_char, end_char, label)
    """
    pairs: List[Tuple[str, str, int, int, int]] = []

    for path in [lex_mturk_path, benchls_path]:
        for sentence, target, cands in _load_file(path):
            # Locate target word in sentence
            match = re.search(r'\b' + re.escape(target) + r'\b',
                              sentence, re.IGNORECASE)
            if not match:
                continue
            sc, ec = match.start(), match.end()

            # Target word → COMPLEX
            pairs.append((sentence, target, sc, ec, 1))

            # Top substitute → SIMPLE (use substituted sentence)
            if cands:
                best_sub = cands[0][0]
                sub_sentence = sentence[:sc] + best_sub + sentence[ec:]
                sub_match = re.search(r'\b' + re.escape(best_sub) + r'\b',
                                      sub_sentence, re.IGNORECASE)
                if sub_match:
                    pairs.append((sub_sentence, best_sub,
                                  sub_match.start(), sub_match.end(), 0))

    print(f"[data_loader] CWI training pairs: {len(pairs)} "
          f"({sum(1 for *_, l in pairs if l == 1)} complex, "
          f"{sum(1 for *_, l in pairs if l == 0)} simple).")
    return pairs
