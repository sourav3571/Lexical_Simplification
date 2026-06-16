# cwi_data.py
"""
CWI Dataset Preparation Module.

Supported sources:
  1. CWI 2018 Shared Task   (primary, ~27k annotated tokens)
  2. LexMTurk               (secondary,  ~500 sentences)
  3. BenchLS                (secondary,  ~929 sentences)

Output format (all sources unified):
    List of CWISample(sentence, target_word, start_char, end_char, label)

CWI 2018 download instructions
────────────────────────────────
Option A — HuggingFace datasets (automatic):
    pip install datasets
    (this module auto-detects and downloads via HuggingFace)

Option B — Manual download:
    Download from: https://zenodo.org/record/1172640
    Extract to the directory set in CWIConfig.cwi_2018_train_dir
    Expected file names:
        english_training_data/WikiNews_Train.tsv
        english_training_data/Wikipedia_Train.tsv
        english_training_data/News_Train.tsv
        english_testing_data/WikiNews_Test.tsv
        english_testing_data/Wikipedia_Test.tsv
        english_testing_data/News_Test.tsv

CWI 2018 TSV format (each row):
    ID  word  sentence  offset_start  offset_end  native_annots
        non_native_annots  native_complex  non_native_complex  binary  prob
    binary column (index 9): 0 = simple, 1 = complex  ← we use this
"""

from __future__ import annotations

import os
import re
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import Counter


@dataclass
class CWISample:
    sentence:   str
    word:       str
    start_char: int
    end_char:   int
    label:      int   # 1 = complex, 0 = simple

    def __repr__(self):
        tag = "COMPLEX" if self.label == 1 else "simple"
        return f"CWISample('{self.word}' [{tag}] in '{self.sentence[:40]}...')"


# ─────────────────────────────────────────────────────────────────────────────
# CWI 2018 Parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cwi2018_tsv(path: str) -> List[CWISample]:
    """
    Parse one CWI 2018 .tsv file.

    Column layout (space-separated, not tab in some files):
        0: ID
        1: sent_id
        2: token  (the target word)
        3: offset_start
        4: offset_end
        5: native_annotators
        6: non_native_annotators
        7: native_complex  (count who said complex)
        8: non_native_complex
        9: binary  ← our label
        10: probabilistic
    """
    samples: List[CWISample] = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Handle both tab and space delimiters
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) < 10:
                continue
            try:
                word       = parts[2].strip()
                sentence   = parts[1].strip() if len(parts) > 11 else parts[3].strip()
                # Rebuild sentence from offset
                try:
                    start = int(parts[3])
                    end   = int(parts[4])
                except (ValueError, IndexError):
                    continue
                binary = int(parts[9])
                label  = 1 if binary == 1 else 0
                samples.append(CWISample(sentence, word, start, end, label))
            except (IndexError, ValueError):
                continue
    return samples


def _parse_cwi2018_simple_format(path: str) -> List[CWISample]:
    """
    Parse CWI 2018 files where the format is:
        sentence TAB word TAB start TAB end TAB label
    (simplified / re-exported format sometimes used)
    """
    samples: List[CWISample] = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            try:
                sentence = parts[0].strip()
                word     = parts[1].strip()
                start    = int(parts[2])
                end      = int(parts[3])
                label    = int(parts[4])
                if label in (0, 1):
                    samples.append(CWISample(sentence, word, start, end, label))
            except (ValueError, IndexError):
                continue
    return samples


def load_cwi_2018(data_dir: str = "cwi2018_data") -> List[CWISample]:
    """
    Load CWI 2018 Shared Task data from a local directory, OR
    automatically download via HuggingFace datasets if available.

    Returns list of CWISample objects.
    """
    # Try HuggingFace datasets first (easiest path)
    try:
        from datasets import load_dataset
        print("[cwi_data] Attempting CWI 2018 load via HuggingFace datasets...")
        ds = load_dataset("AyoubChLin/CWI_korpus", trust_remote_code=True)
        samples: List[CWISample] = []
        for split in ('train', 'validation', 'test'):
            if split not in ds:
                continue
            for row in ds[split]:
                try:
                    sent  = row.get('sentence', row.get('text', ''))
                    word  = row.get('token',    row.get('word', ''))
                    start = int(row.get('start_offset', row.get('offset', 0)))
                    end   = int(row.get('end_offset', start + len(word)))
                    label = int(row.get('complex', row.get('label', 0)))
                    if sent and word:
                        samples.append(CWISample(sent, word, start, end, label))
                except (KeyError, ValueError, TypeError):
                    continue
        if samples:
            print(f"[cwi_data] HuggingFace: loaded {len(samples)} CWI 2018 samples.")
            return samples
    except Exception as e:
        safe_err = str(e).encode('ascii', 'ignore').decode('ascii')
        print(f"[cwi_data] HuggingFace load failed ({safe_err}). Trying local files...")

    # Fall back to local directory
    if not os.path.isdir(data_dir):
        print(f"[cwi_data] WARNING: '{data_dir}' not found. "
              f"Download CWI 2018 from https://zenodo.org/record/1172640 "
              f"and extract to '{data_dir}/'")
        return []

    samples: List[CWISample] = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith('.tsv'):
                continue
            path = os.path.join(root, fname)
            new  = _parse_cwi2018_tsv(path)
            if not new:
                new = _parse_cwi2018_simple_format(path)
            samples.extend(new)
            print(f"[cwi_data] {fname}: {len(new)} samples")

    print(f"[cwi_data] CWI 2018 total: {len(samples)} samples.")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# LexMTurk / BenchLS loader (produces CWISample objects)
# ─────────────────────────────────────────────────────────────────────────────

def _load_lexical_dataset(path: str) -> List[CWISample]:
    """
    Load LexMTurk or BenchLS and produce CWISample pairs:
      - Target word → complex (label=1)
      - Top-voted substitute → simple (label=0)
    """
    if not os.path.exists(path):
        print(f"[cwi_data] Not found: {path}")
        return []

    samples: List[CWISample] = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            sentence = parts[0].strip().strip('"')
            target   = parts[1].strip().lower()
            raw      = parts[3:] if (len(parts) > 3 and parts[2].isdigit()) else parts[2:]

            # Parse candidates
            has_colons = any(':' in p for p in raw)
            cands: List[Tuple[str, int]] = []
            if has_colons:
                for item in raw:
                    if ':' in item:
                        tok = item.split(':')
                        try:
                            # tok[0] is rank (e.g. '1'), tok[1] is word (e.g. 'parts')
                            cands.append((tok[1].strip().lower(), int(tok[0])))
                        except (IndexError, ValueError):
                            pass
            else:
                cnt = Counter(p.strip().lower() for p in raw if p.strip().isalpha())
                cands = list(cnt.items())

            if not cands:
                continue

            # Target word → COMPLEX
            m = re.search(r'\b' + re.escape(target) + r'\b', sentence, re.IGNORECASE)
            if m:
                samples.append(CWISample(sentence, target, m.start(), m.end(), 1))

            # Top substitute → SIMPLE
            if has_colons:
                best_sub = min(cands, key=lambda x: x[1])[0]
            else:
                best_sub = max(cands, key=lambda x: x[1])[0]
            if best_sub != target and m:
                sub_sent  = sentence[:m.start()] + best_sub + sentence[m.end():]
                sub_match = re.search(r'\b' + re.escape(best_sub) + r'\b',
                                      sub_sent, re.IGNORECASE)
                if sub_match:
                    samples.append(CWISample(
                        sub_sent, best_sub,
                        sub_match.start(), sub_match.end(), 0))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Dataset combiner + splitter
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_dataset(
    cwi_2018_dir:   str = "cwi2018_data",
    lex_mturk_path: str = "lex_mturk.txt",
    benchls_path:   str = "BenchLS.txt",
    val_split:      float = 0.10,
    test_split:     float = 0.10,
    seed:           int   = 42,
) -> Tuple[List[CWISample], List[CWISample], List[CWISample]]:
    """
    Build train / val / test splits from all three sources.

    Returns: (train_samples, val_samples, test_samples)

    Strategy:
      - CWI 2018 data is split 80/10/10
      - LexMTurk + BenchLS go entirely into train (too small for reliable val/test)
      - All lists are shuffled with fixed seed for reproducibility
    """
    print("\n[cwi_data] Loading all datasets...")

    cwi_samples = load_cwi_2018(cwi_2018_dir)
    lex_samples = _load_lexical_dataset(lex_mturk_path)
    bls_samples = _load_lexical_dataset(benchls_path)

    print(f"[cwi_data] CWI 2018: {len(cwi_samples)} samples "
          f"({sum(s.label for s in cwi_samples)} complex, "
          f"{sum(1-s.label for s in cwi_samples)} simple)")
    print(f"[cwi_data] LexMTurk: {len(lex_samples)} samples")
    print(f"[cwi_data] BenchLS:  {len(bls_samples)} samples")

    # Split CWI 2018 if present, otherwise split LexMTurk + BenchLS
    random.seed(seed)
    if not cwi_samples:
        print("[cwi_data] CWI 2018 is empty. Splitting LexMTurk + BenchLS into train/val/test splits.")
        extra = lex_samples + bls_samples
        random.shuffle(extra)
        n_total = len(extra)
        n_test  = int(n_total * test_split)
        n_val   = int(n_total * val_split)
        test_s  = extra[:n_test]
        val_s   = extra[n_test: n_test + n_val]
        train_s = extra[n_test + n_val:]
    else:
        random.shuffle(cwi_samples)
        n_total = len(cwi_samples)
        n_test  = int(n_total * test_split)
        n_val   = int(n_total * val_split)
        test_s  = cwi_samples[:n_test]
        val_s   = cwi_samples[n_test: n_test + n_val]
        train_s = cwi_samples[n_test + n_val:]

        # Add LexMTurk + BenchLS to train only
        extra = lex_samples + bls_samples
        random.shuffle(extra)
        train_s = train_s + extra

    random.shuffle(train_s)
    print(f"\n[cwi_data] Split summary:")
    print(f"  Train: {len(train_s)} samples")
    print(f"  Val:   {len(val_s)} samples")
    print(f"  Test:  {len(test_s)} samples")
    _print_class_balance("Train", train_s)
    _print_class_balance("Val",   val_s)
    _print_class_balance("Test",  test_s)
    return train_s, val_s, test_s


def _print_class_balance(name: str, samples: List[CWISample]) -> None:
    if not samples:
        return
    n = len(samples)
    nc = sum(s.label for s in samples)
    ns = n - nc
    print(f"    {name}: {nc} complex ({100*nc/n:.1f}%), "
          f"{ns} simple ({100*ns/n:.1f}%)")
