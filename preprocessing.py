import spacy
from typing import List, Dict, Any

class Preprocessor:
    """
    Preprocessor class handles tokenization, POS tagging, lemmatization,
    and filtering of structural/grammatical words using spaCy.
    """
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """
        Initializes the spaCy language model.
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # Fallback if model is not downloaded; try to download it
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)

    def preprocess(self, sentence: str) -> List[Dict[str, Any]]:
        """
        Tokenizes the input sentence, performs POS tagging, lemmatization,
        and flags words to ignore based on grammatical roles.
        """
        if not sentence.strip():
            return []
            
        doc = self.nlp(sentence)
        tokens_info: List[Dict[str, Any]] = []
        
        # POS categories to skip in lexical simplification
        skip_pos = {
            'DET', 'PRON', 'ADP', 'CCONJ', 'SCONJ', 
            'PUNCT', 'NUM', 'PROPN', 'AUX', 'PART', 'SYM', 'SPACE'
        }
        
        for token in doc:
            word_lower = token.text.strip().lower()
            if not word_lower or not token.text.isalpha():
                continue
                
            is_skippable = token.pos_ in skip_pos
            
            tokens_info.append({
                'text': token.text,
                'lemma': token.lemma_.lower(),
                'pos': token.pos_,
                'start': token.idx,
                'end': token.idx + len(token.text),
                'is_skippable': is_skippable
            })
            
        return tokens_info
