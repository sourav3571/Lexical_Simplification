# dynamic_cwi.py

import numpy as np
import wordfreq
from typing import List, Tuple, Dict, Any
from bert_complexity import BERTComplexityScorer


class DynamicContextualCWI:
    """
    DynamicContextualCWI evaluates complexity scores for all content words in a sentence.

    A word is flagged COMPLEX if a frequency-adjusted complexity score exceeds the
    CWI threshold. This better reflects standard CWI practice by making low-
    frequency words easier to flag while requiring a stronger signal for more
    common vocabulary.
    """

    FREQ_SIMPLE_FLOOR = 4.5  # Zipf freq threshold for neutral complexity adjustment
    MAX_FREQ_BONUS = 0.10    # Max score boost for rare words
    MAX_FREQ_PENALTY = 0.05  # Max penalty for frequent words

    def __init__(self, tokenizer=None, model=None, bert_model=None, device=None) -> None:
        self.scorer = BERTComplexityScorer(tokenizer, model, bert_model, device)

    def _adjusted_complexity_score(self, score: float, zipf: float) -> float:
        """
        Apply a light frequency-based adjustment to the raw complexity score.

        Rare words receive a small bonus, while common words receive a small
        penalty so only sufficiently difficult common words are marked complex.
        """
        bonus = max(0.0, min(self.MAX_FREQ_BONUS, (self.FREQ_SIMPLE_FLOOR - zipf) / 10.0))
        penalty = max(0.0, min(self.MAX_FREQ_PENALTY, (zipf - self.FREQ_SIMPLE_FLOOR) / 20.0))
        return score + bonus - penalty

    def identify_complex_words(
        self,
        sentence: str,
        content_tokens: List[Tuple[str, str, int, int]],
        cwi_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """
        Parameters
        ----------
        content_tokens : list of (word, pos, start_char, end_char)
        cwi_threshold  : minimum complexity score to flag a word as complex

        Returns
        -------
        List of dicts with keys: word, pos, start_char, end_char, score,
                                  is_complex, mean, std
        """
        if not content_tokens:
            return []

        # 1. Compute complexity score for every content word
        results = []
        scores  = []
        for word, pos, start_char, end_char in content_tokens:
            score = self.scorer.compute_complexity_score(
                sentence, word, start_char, end_char, pos)
            scores.append(score)
            results.append({
                'word':       word,
                'pos':        pos,
                'start_char': start_char,
                'end_char':   end_char,
                'score':      score
            })

        # 2. Compute sentence-level statistics (for diagnostic display)
        scores_arr = np.array(scores)
        mean_score = float(np.mean(scores_arr)) if len(scores) > 0 else 0.0
        std_score  = float(np.std(scores_arr))  if len(scores) > 0 else 0.0

        # 3. Apply frequency-aware complexity threshold
        for res in results:
            word_zipf = wordfreq.zipf_frequency(res['word'].lower(), 'en')
            adjusted_score = self._adjusted_complexity_score(res['score'], word_zipf)
            is_comp = adjusted_score >= cwi_threshold
            res['adjusted_score'] = adjusted_score
            res['word_zipf'] = word_zipf
            res['is_complex'] = is_comp
            res['mean']       = mean_score
            res['std']        = std_score

        return results
