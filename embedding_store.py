# embedding_store.py
"""
Unified GloVe + FastText loader with nearest-neighbour retrieval.

Usage:
    store = EmbeddingStore(glove_path='glove.840B.300d.txt',
                           fasttext_path='crawl-300d-2M.vec')
    neighbours = store.nearest_neighbours('physician', topn=20, source='glove')
    # → ['doctor', 'surgeon', 'clinician', ...]

Both models are optional. If a path is missing the source is silently
skipped and candidates fall through to the next priority tier.
"""

import os
import numpy as np
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight vector store (no gensim dependency for raw .txt files)
# ─────────────────────────────────────────────────────────────────────────────

class _VectorStore:
    """Memory-efficient word-vector store loaded from a plain-text .vec file."""

    def __init__(self, path: str, max_words: int = 500_000) -> None:
        self.words:   List[str]      = []
        self.vectors: np.ndarray     = np.array([])
        self._index:  dict           = {}
        self._loaded: bool           = False

        if not path or not os.path.exists(path):
            print(f"[EmbeddingStore] Path not found, skipping: {path}")
            return

        print(f"[EmbeddingStore] Loading {path} (max {max_words} words)…")
        words, vecs = [], []
        with open(path, encoding='utf-8', errors='ignore') as f:
            first = f.readline().strip().split()
            # Skip header line if it's "vocab_size dim" (FastText style)
            if len(first) == 2 and first[0].isdigit():
                pass   # skip
            else:
                # GloVe: no header, first line is a word vector
                tok = first
                if len(tok) > 2:
                    words.append(tok[0].lower())
                    vecs.append(np.array(tok[1:], dtype=np.float32))

            for i, line in enumerate(f):
                if i >= max_words:
                    break
                tok = line.rstrip().split(' ')
                if len(tok) < 10:
                    continue
                words.append(tok[0].lower())
                vecs.append(np.array(tok[1:], dtype=np.float32))

        self.words   = words
        self.vectors = np.stack(vecs)              # (N, D)
        # L2-normalise for fast cosine via dot product
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = self.vectors / norms
        self._index  = {w: i for i, w in enumerate(words)}
        self._loaded = True
        print(f"[EmbeddingStore] Loaded {len(self.words):,} words.")

    @property
    def loaded(self) -> bool:
        return self._loaded

    def __contains__(self, word: str) -> bool:
        return word.lower() in self._index

    def __getitem__(self, word: str) -> Optional[np.ndarray]:
        idx = self._index.get(word.lower())
        return self.vectors[idx] if idx is not None else None

    def nearest(self, word: str, topn: int = 20) -> List[Tuple[str, float]]:
        """Return (word, cosine_similarity) pairs sorted descending."""
        vec = self[word]
        if vec is None:
            return []
        sims   = self.vectors @ vec          # dot product on L2-normed = cosine
        top_i  = np.argpartition(sims, -topn - 1)[-topn - 1:]
        top_i  = top_i[np.argsort(-sims[top_i])]
        results = []
        for i in top_i:
            w = self.words[i]
            if w != word.lower():
                results.append((w, float(sims[i])))
        return results[:topn]


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingStore:
    """
    Wraps GloVe and FastText vector stores behind a unified API.

    Parameters
    ----------
    glove_path    : Path to GloVe 840B 300d plain-text file (optional).
    fasttext_path : Path to FastText crawl-300d-2M plain-text file (optional).
    max_words     : Maximum vocabulary to load per model (default 500 k).
                    Reduce to 200_000 if RAM is constrained.

    Also supports gensim KeyedVectors if you pass a gensim model via
    glove_model / fasttext_model keyword arguments.
    """

    def __init__(
        self,
        glove_path:    Optional[str] = None,
        fasttext_path: Optional[str] = None,
        max_words:     int = 500_000,
        glove_model=None,       # pre-loaded gensim KeyedVectors
        fasttext_model=None,
    ) -> None:
        # If gensim models passed directly, wrap them
        self._glove_gensim    = glove_model
        self._fasttext_gensim = fasttext_model

        # Otherwise load from plain-text files
        self._glove    = _VectorStore(glove_path,    max_words) if glove_path    else None
        self._fasttext = _VectorStore(fasttext_path, max_words) if fasttext_path else None

    # ── Internal: pick the right backend ──────────────────────────────────────

    def _nearest_gensim(self, model, word: str, topn: int) -> List[Tuple[str, float]]:
        """Delegate to a gensim KeyedVectors object."""
        if model is None:
            return []
        word_l = word.lower()
        key = word_l if word_l in model else (word if word in model else None)
        if key is None:
            return []
        try:
            return [(w.lower(), float(s)) for w, s in model.most_similar(key, topn=topn)]
        except Exception:
            return []

    # ── Public API ────────────────────────────────────────────────────────────

    def nearest_neighbours(
        self,
        word:   str,
        topn:   int = 20,
        source: str = 'glove'   # 'glove' | 'fasttext' | 'both'
    ) -> List[Tuple[str, float]]:
        """
        Return [(neighbour, cosine_similarity), …] ranked by similarity.

        source='both' merges GloVe and FastText results (deduplicated,
        max cosine sim kept when a word appears in both).
        """
        word = word.lower()
        results: dict = {}

        def _merge(pairs):
            for w, s in pairs:
                if w not in results or results[w] < s:
                    results[w] = s

        if source in ('glove', 'both'):
            if self._glove_gensim:
                _merge(self._nearest_gensim(self._glove_gensim, word, topn))
            elif self._glove and self._glove.loaded:
                _merge(self._glove.nearest(word, topn))

        if source in ('fasttext', 'both'):
            if self._fasttext_gensim:
                _merge(self._nearest_gensim(self._fasttext_gensim, word, topn))
            elif self._fasttext and self._fasttext.loaded:
                _merge(self._fasttext.nearest(word, topn))

        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return ranked[:topn]

    def similarity(self, word1: str, word2: str, source: str = 'glove') -> float:
        """Cosine similarity between two words. Returns 0.0 if either is OOV."""
        w1, w2 = word1.lower(), word2.lower()

        # Gensim path
        model = self._glove_gensim if source == 'glove' else self._fasttext_gensim
        if model is not None:
            try:
                return float(model.similarity(w1, w2))
            except Exception:
                return 0.0

        # Plain-text path
        store = self._glove if source == 'glove' else self._fasttext
        if store is None or not store.loaded:
            return 0.0
        v1, v2 = store[w1], store[w2]
        if v1 is None or v2 is None:
            return 0.0
        return float(np.dot(v1, v2))   # already L2-normalised

    def __contains__(self, word: str) -> bool:
        word = word.lower()
        if self._glove_gensim and word in self._glove_gensim:
            return True
        if self._fasttext_gensim and word in self._fasttext_gensim:
            return True
        if self._glove and self._glove.loaded and word in self._glove:
            return True
        if self._fasttext and self._fasttext.loaded and word in self._fasttext:
            return True
        return False
