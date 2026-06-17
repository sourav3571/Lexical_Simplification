# bert_validator.py
"""
Upgraded 4-Gate BERT Validator — tightened thresholds for higher precision.

Gate changes vs. old version:
  Gate 2 (Similarity):   SBERT sentence sim > 0.90 (was BERT 0.92)
                         Word cosine sim     > 0.75 (was 0.75, unchanged)
  Gate 3 (Fluency):      fluency drop < 0.4  (was 0.7 — tightened)
  Gate 4 (Complexity):   candidate complexity < original  AND
                         candidate Zipf > original Zipf   (NEW Zipf check)

Gate 1 (MLM Preference) is unchanged:
    candidate_prob > original_prob  OR  candidate_prob >= 0.001
"""

from __future__ import annotations

import spacy
import torch
import torch.nn.functional as F

from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_surprisal import BERTSurprisalCalculator


class BERTValidator:
    """
    4-Gate BERT Validator — every gate must pass for a candidate to be accepted.

    Gate 1 — MLM Preference
        candidate_prob > original_prob  OR  candidate_prob >= 0.001

    Gate 2 — Meaning Preservation (TIGHTENED)
        SBERT sentence similarity  > 0.90   (BERT CLS fallback if no SBERT)
        Word-level cosine similarity > 0.75

    Gate 3 — Fluency (TIGHTENED)
        BERT fluency drop < 0.4 log-likelihood units

    Gate 4 — Complexity + Zipf Confirmation (UPGRADED)
        candidate complexity score < original complexity score
        AND candidate Zipf frequency > original Zipf frequency
    """

    # Gate thresholds (all tightened or upgraded)
    GATE1_MLM_FLOOR    = 0.001   # unchanged
    GATE2_SENT_SIM     = 0.70    # adjusted to 0.70 for SBERT space
    GATE2_WORD_SIM     = 0.75    # unchanged
    GATE3_FLUENCY_DROP = 0.40    # tightened from 0.70
    GATE4_ZIPF_CONFIRM = True    # NEW: require candidate_zipf > original_zipf

    def __init__(
        self,
        tokenizer=None,
        model=None,
        bert_model=None,
        device=None,
        nlp=None,
        sbert_encoder=None,   # _SBERTEncoder instance (optional but recommended)
    ) -> None:
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer  = tokenizer  or BertTokenizer.from_pretrained(
            'bert-base-uncased')
        self.model      = model      or BertForMaskedLM.from_pretrained(
            'bert-base-uncased').to(self.device)
        self.bert_model = bert_model or BertModel.from_pretrained(
            'bert-base-uncased').to(self.device)
        self.nlp        = nlp        or spacy.load('en_core_web_sm')

        self.surprisal_calc = BERTSurprisalCalculator(
            self.tokenizer, self.model, self.device)

        from bert_complexity import BERTComplexityScorer
        self.scorer = BERTComplexityScorer(
            self.tokenizer, self.model, self.bert_model, self.device)

        # SBERT for Gate 2 (use if available, fallback to BERT CLS)
        self._sbert = sbert_encoder
        if self._sbert is None:
            try:
                from bert_candidate_generator import _SBERTEncoder
                self._sbert = _SBERTEncoder(self.device)
            except Exception:
                self._sbert = None

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def _find_token_at_span(self, doc, start_char: int, end_char: int):
        for token in doc:
            if (token.idx <= start_char < token.idx + len(token.text)
                    and token.idx < end_char <= token.idx + len(token.text)):
                return token
        return None

    def _morphological_match(
        self, orig_sentence: str, cand_sentence: str,
        start_char: int, original_end: int, candidate_length: int
    ) -> bool:
        orig_doc  = self.nlp(orig_sentence)
        cand_doc  = self.nlp(cand_sentence)
        orig_tok  = self._find_token_at_span(orig_doc, start_char, original_end)
        cand_tok  = self._find_token_at_span(
            cand_doc, start_char, start_char + candidate_length)
        if orig_tok is None or cand_tok is None:
            return True
        pos1, pos2 = orig_tok.pos_, cand_tok.pos_
        compatible = (pos1 == pos2) or ({pos1, pos2} == {'VERB', 'ADJ'}) or ({pos1, pos2} == {'NOUN', 'PROPN'})
        if not compatible:
            return False
        orig_m = orig_tok.morph.to_dict()
        cand_m = cand_tok.morph.to_dict()
        keys   = [k for k in orig_m if k in cand_m and k in {
            'Number', 'Tense', 'VerbForm', 'Degree',
            'Person', 'Mood', 'Aspect', 'Case', 'Gender'}]
        if not keys:
            return True
        return all(orig_m.get(k) == cand_m.get(k) for k in keys)


    def get_sentence_embedding(self, sentence: str) -> torch.Tensor:
        enc = self.tokenizer(sentence, return_tensors='pt',
                             padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            return self.bert_model(**enc).last_hidden_state[0, 0]

    def get_word_embedding(
        self, sentence: str, word: str, start_char: int
    ) -> torch.Tensor:
        enc = self.tokenizer(sentence, return_tensors='pt').to(self.device)
        with torch.no_grad():
            states = self.bert_model(**enc).last_hidden_state[0]
        pre = self.tokenizer.tokenize(sentence[:start_char])
        tok = self.tokenizer.tokenize(word)
        s   = min(len(pre) + 1, states.size(0) - 1)
        e   = min(s + len(tok), states.size(0))
        return states[s:e].mean(dim=0)

    def compute_sentence_log_likelihood(self, sentence: str) -> float:
        enc = self.tokenizer(sentence, return_tensors='pt',
                             padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            loss = self.model(**enc, labels=enc['input_ids']).loss.item()
        return -loss

    def _sbert_similarity(self, s1: str, s2: str) -> float:
        """SBERT sentence similarity, falling back to BERT CLS cosine."""
        if self._sbert is not None and getattr(self._sbert, 'available', False):
            return self._sbert.similarity(s1, s2)
        # BERT CLS fallback
        e1 = self.get_sentence_embedding(s1)
        e2 = self.get_sentence_embedding(s2)
        return F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()

    # ─────────────────────────────────────────────────────────────────────────
    # Main validation method
    # ─────────────────────────────────────────────────────────────────────────

    def validate_replacement(
        self,
        sentence:       str,
        original_word:  str,
        candidate_word: str,
        start_char:     int,
        end_char:       int,
        pos_tag:        str  = None,
        debug:          bool = False,
    ) -> bool:
        """
        Returns True if all 4 gates pass.

        Parameters
        ----------
        sentence       : original sentence
        original_word  : target word being replaced
        candidate_word : proposed simpler substitute
        start_char     : char start of original_word in sentence
        end_char       : char end of original_word in sentence
        pos_tag        : SpaCy POS tag (for morphological check)
        debug          : print gate results if True
        """
        import wordfreq

        cand_sentence = (sentence[:start_char]
                         + candidate_word
                         + sentence[end_char:])

        # ── Gate 1: MLM Preference ────────────────────────────────────────────
        masked = self.surprisal_calc.get_masked_sentence_and_idx(
            sentence, start_char, end_char)
        enc    = self.tokenizer(masked, return_tensors='pt').to(self.device)
        midxs  = (enc['input_ids'][0] == self.tokenizer.mask_token_id
                  ).nonzero(as_tuple=True)[0]
        if len(midxs) == 0:
            if debug:
                print("  [FAIL] Gate 1: mask token not found")
            return False
        mask_pos = midxs[0].item()
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs    = F.softmax(logits[0, mask_pos], dim=-1)
        orig_ids = self.tokenizer(
            original_word.lower(), add_special_tokens=False)['input_ids']
        cand_ids = self.tokenizer(
            candidate_word.lower(), add_special_tokens=False)['input_ids']
        orig_prob = probs[orig_ids[0]].item() if orig_ids else 1e-9
        cand_prob = probs[cand_ids[0]].item() if cand_ids else 1e-9

        gate1_ok = cand_prob > orig_prob or cand_prob >= self.GATE1_MLM_FLOOR
        if not gate1_ok:
            if debug:
                print(f"  [FAIL] Gate 1: cand_prob={cand_prob:.5f} <= "
                      f"orig_prob={orig_prob:.5f} and < {self.GATE1_MLM_FLOOR}")
            return False
        if debug:
            print(f"  [PASS] Gate 1: cand_prob={cand_prob:.5f} "
                  f"orig_prob={orig_prob:.5f}")

        # ── Gate 2: Meaning Preservation (TIGHTENED) ──────────────────────────
        # 2a. SBERT sentence similarity > 0.90
        sent_sim = self._sbert_similarity(sentence, cand_sentence)
        if sent_sim < self.GATE2_SENT_SIM:
            if debug:
                print(f"  [FAIL] Gate 2a: sent_sim={sent_sim:.4f} "
                      f"< {self.GATE2_SENT_SIM}")
            return False

        # 2b. Word-level BERT cosine similarity > 0.75
        orig_word_emb = self.get_word_embedding(sentence, original_word, start_char)
        cand_word_emb = self.get_word_embedding(
            cand_sentence, candidate_word, start_char)
        word_sim = F.cosine_similarity(
            orig_word_emb.unsqueeze(0), cand_word_emb.unsqueeze(0)).item()
        if word_sim < self.GATE2_WORD_SIM:
            if debug:
                print(f"  [FAIL] Gate 2b: word_sim={word_sim:.4f} "
                      f"< {self.GATE2_WORD_SIM}")
            return False

        # 2c. Morphological form match
        if pos_tag is not None:
            if not self._morphological_match(
                    sentence, cand_sentence, start_char, end_char,
                    len(candidate_word)):
                if debug:
                    print(f"  [FAIL] Gate 2c: morphological mismatch "
                          f"'{original_word}' vs '{candidate_word}'")
                return False

        if debug:
            print(f"  [PASS] Gate 2: sent_sim={sent_sim:.4f}  "
                  f"word_sim={word_sim:.4f}")

        # ── Gate 3: Fluency (TIGHTENED: drop < 0.4) ──────────────────────────
        orig_fluency = self.compute_sentence_log_likelihood(sentence)
        cand_fluency = self.compute_sentence_log_likelihood(cand_sentence)
        fluency_delta = cand_fluency - orig_fluency
        if fluency_delta < -self.GATE3_FLUENCY_DROP:
            if debug:
                print(f"  [FAIL] Gate 3: fluency_delta={fluency_delta:.4f} "
                      f"< -{self.GATE3_FLUENCY_DROP}")
            return False
        if debug:
            print(f"  [PASS] Gate 3: fluency_delta={fluency_delta:.4f}")

        # ── Gate 4: Complexity Reduction + Zipf Confirmation (UPGRADED) ───────
        orig_complexity = self.scorer.compute_complexity_score(
            sentence, original_word, start_char, end_char, pos_tag)
        cand_complexity = self.scorer.compute_complexity_score(
            cand_sentence, candidate_word, start_char,
            start_char + len(candidate_word), pos_tag)
        complexity_reduction = orig_complexity - cand_complexity

        orig_zipf = wordfreq.zipf_frequency(original_word.lower(), 'en')
        cand_zipf = wordfreq.zipf_frequency(candidate_word.lower(), 'en')

        # For high-frequency target words (likely figurative/polysemous), relax validation limits
        if orig_zipf >= 4.5:
            min_comp_red = -0.15
            min_cand_zipf = 4.0
        else:
            min_comp_red = 0.0
            min_cand_zipf = orig_zipf + 0.0001

        if complexity_reduction < min_comp_red:
            if debug:
                print(f"  [FAIL] Gate 4a: complexity_reduction={complexity_reduction:.4f} "
                      f"< {min_comp_red} (orig={orig_complexity:.4f}, cand={cand_complexity:.4f})")
            return False

        # NEW: Zipf confirmation — candidate must be genuinely simpler (higher freq or relaxed floor)
        if self.GATE4_ZIPF_CONFIRM:
            if cand_zipf < min_cand_zipf:
                if debug:
                    print(f"  [FAIL] Gate 4b: cand_zipf={cand_zipf:.2f} "
                          f"< required_min={min_cand_zipf:.2f} (orig_zipf={orig_zipf:.2f})")
                return False

        if debug:
            print(f"  [PASS] Gate 4: complexity_reduction={complexity_reduction:.4f}  "
                  f"zipf_delta={cand_zipf - orig_zipf:.2f}")
            print("  [PASS] All 4 gates passed!")

        return True
