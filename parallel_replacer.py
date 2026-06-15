# parallel_replacer.py

import re
from typing import Dict

class ParallelReplacer:
    """
    ParallelReplacer applies all target replacements simultaneously
    on the original sentence, preventing cascading substitution errors.
    """
    def __init__(self) -> None:
        pass

    def replace_all(self, sentence: str, replacements: Dict[str, str]) -> str:
        """
        Applies replacements to the original sentence at once using regex
        word boundaries, preserving original sentence context for all decisions.
        """
        final_sentence = sentence
        for word, replacement in replacements.items():
            final_sentence = re.sub(
                r'\b' + re.escape(word) + r'\b',
                replacement,
                final_sentence,
                flags=re.IGNORECASE
            )
        return final_sentence
