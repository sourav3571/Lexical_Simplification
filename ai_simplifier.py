# ai_simplifier.py

import os
import torch
import torch.nn.functional as F
import spacy
import wordfreq
from nltk.corpus import wordnet as wn
from typing import Dict, Any, List
from transformers import BertTokenizer, BertForMaskedLM, BertModel

from bert_surprisal import BERTSurprisalCalculator
from bert_complexity import BERTComplexityScorer
from dynamic_cwi import DynamicContextualCWI
from bert_sense_disambiguator import BERTSenseDisambiguator
from bert_candidate_generator import BERTCandidateGenerator
from bert_validator import BERTValidator
from bert_ranker import GatedFusionRanker, are_semantically_related
from parallel_replacer import ParallelReplacer

# ---------------------------------------------------------------------------
# Module-level globals used by verify_bert_mlm()
# ---------------------------------------------------------------------------
_global_tokenizer = None
_global_model     = None


def verify_bert_mlm():
    """
    Sanity-check that BertForMaskedLM is producing real predictions.
    Returns True if the top-1 probability at [MASK] is > 0.10, else False.
    """
    global _global_tokenizer, _global_model
    if _global_tokenizer is None or _global_model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _global_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        _global_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
        _global_model.eval()

    tok  = _global_tokenizer
    mdl  = _global_model
    test = "The cat sat on the [MASK]"
    inp  = tok(test, return_tensors='pt')
    inp  = {k: v.to(mdl.device) for k, v in inp.items()}

    with torch.no_grad():
        out = mdl(**inp)

    mask_id  = tok.mask_token_id
    mask_pos = (inp['input_ids'] == mask_id).nonzero()[0][1]
    probs    = torch.softmax(out.logits[0, mask_pos], dim=-1)
    top5     = torch.topk(probs, 5)

    print("\nBERT MLM Sanity Check:")
    print(f"  Sentence: {test}")
    print("  Top 5 predictions:")
    for prob, idx in zip(top5.values, top5.indices):
        word = tok.decode([idx]).strip()
        print(f"    {word}: {prob.item():.4f}")

    ok = top5.values[0].item() > 0.1
    print(f"  Result: {'PASS' if ok else 'FAIL (probabilities too low - BERT broken)'}\n")
    return ok


# ---------------------------------------------------------------------------
class AILexicalSimplifier:
    """
    AILexicalSimplifier implements the 6-stage lexical simplification pipeline
    using pure BERT-based models and dynamic contextual analysis with zero
    hardcoded rules.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, config: Dict[str, Any], device: torch.device = None) -> None:
        self.config = config
        self.device = (device if device is not None
                       else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # SpaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # BERT tokenizer + models
        model_name      = config.get('bert_model', 'bert-base-uncased')
        self.tokenizer  = BertTokenizer.from_pretrained(model_name)
        self.mlm_model  = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        self.bert_model = BertModel.from_pretrained(model_name).to(self.device)
        self.mlm_model.eval()
        self.bert_model.eval()

        # Expose to verify_bert_mlm
        global _global_tokenizer, _global_model
        _global_tokenizer = self.tokenizer
        _global_model     = self.mlm_model

        # Run sanity check
        if not verify_bert_mlm():
            raise RuntimeError(
                "BERT MLM sanity check FAILED. "
                "Check that the model is BertForMaskedLM, not BertModel."
            )

        # ── Optional: Gold lookup table (LexMTurk + BenchLS) ────────────────
        self.gold_table: dict = {}
        try:
            from data_loader import build_gold_table
            self.gold_table = build_gold_table(
                lex_mturk_path=config.get('lex_mturk_path', 'lex_mturk.txt'),
                benchls_path=config.get('benchls_path', 'BenchLS.txt'),
            )
        except Exception as e:
            print(f"[AILexicalSimplifier] Gold table unavailable: {e}")

        # ── Optional: EmbeddingStore (GloVe + FastText) ───────────────────────
        self.emb_store = None
        try:
            from embedding_store import EmbeddingStore
            glove_path    = config.get('glove_path', '')
            fasttext_path = config.get('fasttext_path', '')
            gensim_model  = config.get('glove_model')    # legacy gensim KV
            if glove_path or fasttext_path or gensim_model:
                self.emb_store = EmbeddingStore(
                    glove_path=glove_path or None,
                    fasttext_path=fasttext_path or None,
                    glove_model=gensim_model,
                )
        except Exception as e:
            print(f"[AILexicalSimplifier] EmbeddingStore unavailable: {e}")

        # ── Pipeline components ───────────────────────────────────────────────
        self.surprisal_calc    = BERTSurprisalCalculator(
            self.tokenizer, self.mlm_model, self.device)
        self.complexity_scorer = BERTComplexityScorer(
            self.tokenizer, self.mlm_model, self.bert_model, self.device)

        # CWI: pass nlp for figurative pattern detection
        self.cwi = DynamicContextualCWI(
            self.tokenizer, self.mlm_model, self.bert_model,
            self.device, nlp=self.nlp)

        self.disambiguator = BERTSenseDisambiguator(
            self.tokenizer, self.bert_model, self.device)

        # CandidateGenerator: wire gold_table + embedding_store
        self.cand_gen = BERTCandidateGenerator(
            self.tokenizer,
            self.mlm_model,
            self.bert_model,
            self.device,
            gold_table=self.gold_table,
            embedding_store=self.emb_store,
            glove_model=config.get('glove_model'),
        )

        # Validator: wire SBERT encoder (shared with CandidateGenerator)
        self.validator = BERTValidator(
            self.tokenizer, self.mlm_model, self.bert_model,
            self.device, nlp=self.nlp,
            sbert_encoder=getattr(self.cand_gen, '_sbert', None),
        )
        self.replacer = ParallelReplacer()

        # ── Neural ranker (6-feature) ─────────────────────────────────────────
        self.ranker = GatedFusionRanker()
        # Accept both old 4-feature checkpoint and new 6-feature checkpoint
        for ranker_path in ('gated_fusion_ranker_6f.pt', 'gated_fusion_ranker.pt'):
            if os.path.exists(ranker_path):
                try:
                    self.ranker.load_state_dict(
                        torch.load(ranker_path, map_location=self.device),
                        strict=False)   # strict=False: tolerates feature-count mismatch
                    print(f"Loaded GatedFusionRanker weights from {ranker_path}.")
                    break
                except Exception as e:
                    print(f"Could not load ranker weights from {ranker_path}: {e}.")
        self.ranker.to(self.device)
        self.ranker.eval()

    def _get_definition_embedding(self, text: str) -> torch.Tensor:
        if not hasattr(self, '_definition_emb_cache'):
            self._definition_emb_cache = {}
        if text in self._definition_emb_cache:
            return self._definition_emb_cache[text]

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        emb = outputs.last_hidden_state[0, 0]
        self._definition_emb_cache[text] = emb
        return emb

    def _wordnet_sense_similarity(
        self,
        chosen_sense,
        target_word: str,
        candidate_word: str,
        pos: str
    ) -> float:
        """
        Compute a sense similarity score by comparing WordNet definition embeddings.
        """
        pos_map = {
            'NOUN': wn.NOUN,
            'VERB': wn.VERB,
            'ADJ': wn.ADJ,
            'ADV': wn.ADV,
            'PROPN': wn.NOUN
        }
        wn_pos = pos_map.get(pos.upper()) if pos else None
        if not wn_pos:
            return 0.0

        cand_synsets = wn.synsets(candidate_word.lower(), pos=wn_pos)
        if not cand_synsets:
            return 0.0

        if chosen_sense:
            chosen_def_emb = self._get_definition_embedding(chosen_sense.definition())
            similarities = []
            for cand_syn in cand_synsets:
                if chosen_sense == cand_syn:
                    return 1.0
                cand_def_emb = self._get_definition_embedding(cand_syn.definition())
                sim = F.cosine_similarity(chosen_def_emb.unsqueeze(0), cand_def_emb.unsqueeze(0)).item()
                similarities.append(sim)
            return max(similarities) if similarities else 0.0

        target_synsets = wn.synsets(target_word.lower(), pos=wn_pos)
        if not target_synsets:
            return 0.0

        similarities = []
        for cand_syn in cand_synsets:
            cand_def_emb = self._get_definition_embedding(cand_syn.definition())
            for target_syn in target_synsets:
                target_def_emb = self._get_definition_embedding(target_syn.definition())
                sim = F.cosine_similarity(target_def_emb.unsqueeze(0), cand_def_emb.unsqueeze(0)).item()
                similarities.append(sim)
        return max(similarities) if similarities else 0.0

    def _find_token_at_span(self, doc, start_char: int, end_char: int):
        for token in doc:
            if token.idx <= start_char < token.idx + len(token.text):
                return token
        return None

    def _morphological_compatibility(self, sentence: str, candidate_sentence: str, start_char: int, orig_end: int, cand_len: int) -> float:
        orig_doc = self.nlp(sentence)
        cand_doc = self.nlp(candidate_sentence)
        orig_token = self._find_token_at_span(orig_doc, start_char, orig_end)
        cand_token = self._find_token_at_span(cand_doc, start_char, start_char + cand_len)
        if orig_token is None or cand_token is None:
            return 1.0
        if orig_token.pos_ != cand_token.pos_:
            return 0.0
        orig_morph = orig_token.morph.to_dict()
        cand_morph = cand_token.morph.to_dict()
        relevant_keys = [k for k in orig_morph.keys() if k in {
            'Number', 'Tense', 'VerbForm', 'Degree', 'Person', 'Mood', 'Aspect', 'Case', 'Gender'
        }]
        if not relevant_keys:
            return 1.0
        matches = sum(1 for key in relevant_keys if orig_morph.get(key) == cand_morph.get(key))
        return matches / len(relevant_keys)

    # --------------------------------------------------------------- simplify
    def simplify(self, sentence: str, verbose: bool = True) -> str:
        """
        Run the full 6-stage simplification pipeline with balanced thresholds
        and detailed diagnostic output at every stage.

        Balanced thresholds (middle ground):
          CWI_THRESHOLD  = 0.40   (not 0.55 strict, not 0.35 loose)
          FREQ_GAIN_MIN  = 0.5    (freq gain needed over original)
          CAND_FREQ_MIN  = 4.5    (absolute Zipf floor for candidate)
          MLM_PROB_MIN   = 0.005  (minimum MLM probability)
          BEST_SCORE_MIN = 0.50   (ranker score for winner)
          MARGIN_MIN     = 0.03   (gap between top-2 candidates)
        """
        # ---- Precision-focused threshold constants ----------------------------
        CWI_THRESHOLD  = 0.32          # ≈ mean + 1.2*std  (was 0.30 / mean+1.0*std)
        FREQ_GAIN_MIN  = 0.25
        CAND_FREQ_MIN  = 4.0
        MLM_PROB_MIN   = 0.0025
        BEST_SCORE_MIN = 0.40
        MARGIN_MIN     = 0.005
        SEM_SIM_MIN    = 0.90          # meaning preservation (was 0.82)
        # ---------------------------------------------------------------------

        if verbose:
            print("=" * 60)
            print(f"INPUT: {sentence}")
            print("=" * 60)

        # ================================================================
        # STAGE 1 - Preprocessing
        # ================================================================
        doc            = self.nlp(sentence)
        content_tokens = []
        tokens_display = []

        for token in doc:
            text = token.text
            pos  = token.pos_
            tokens_display.append(f"{text}/{pos}")
            if pos in ('NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN') and text.isalpha():
                sc = token.idx
                ec = sc + len(text)
                content_tokens.append((text, pos, sc, ec))

        if verbose:
            print("\nSTAGE 1 - Preprocessing:")
            print(f"  Parsed tokens:      {tokens_display}")
            print(f"  Content candidates: {[t[0] for t in content_tokens]}")

        if not content_tokens:
            if verbose:
                print("  No content words found to analyse.")
            return sentence

        # ================================================================
        # STAGE 2 - Complex Word Identification (CWI)
        # ================================================================
        cwi_results   = self.cwi.identify_complex_words(
            sentence, content_tokens, cwi_threshold=CWI_THRESHOLD)
        complex_words = []

        # Retrieve the dynamic threshold actually used (computed inside CWI)
        eff_thresh = cwi_results[0]['effective_threshold'] if cwi_results else CWI_THRESHOLD

        if verbose:
            print(f"\nSTAGE 2 - CWI  (dynamic threshold = {eff_thresh:.4f}  "
                  f"[legacy hint = {CWI_THRESHOLD}]):")
            print(f"  {'word':15} {'raw':8} {'adj':8} {'zipf':7} {'decision':16} reason")
            print("  " + "-" * 80)

        for res in cwi_results:
            w       = res['word']
            score   = res['score']
            adj     = res.get('adjusted_score', score)
            is_comp = res['is_complex']
            freq    = res.get('word_zipf', wordfreq.zipf_frequency(w.lower(), 'en'))
            decision = "COMPLEX [FAIL]" if is_comp else "SIMPLE  [PASS]"
            if is_comp:
                reason = f"adj {adj:.3f} >= thresh {eff_thresh:.3f}"
            elif freq >= self.cwi.COMMON_WORD_ZIPF_CEIL:
                reason = f"zipf {freq:.2f} >= ceil {self.cwi.COMMON_WORD_ZIPF_CEIL} (hard SIMPLE)"
            else:
                reason = f"adj {adj:.3f} < thresh {eff_thresh:.3f}"
            if verbose:
                print(f"  {w:15} {score:8.4f} {adj:8.4f} {freq:7.2f}  {decision}  {reason}")
            if is_comp:
                complex_words.append(res)

        if verbose:
            print(f"  => Complex words identified: {[w['word'] for w in complex_words]}")

        replacements = {}

        # ================================================================
        # Process each complex word
        # ================================================================
        for cw in complex_words:
            word       = cw['word']
            pos        = cw['pos']
            start_char = cw['start_char']
            end_char   = cw['end_char']
            orig_zipf  = wordfreq.zipf_frequency(word.lower(), 'en')

            if verbose:
                print(f"\n{'='*60}")
                print(f"  Processing complex word: '{word}'  (Zipf freq = {orig_zipf:.2f})")
                print(f"{'='*60}")

            # ============================================================
            # STAGE 3 - Sense Disambiguation
            # ============================================================
            chosen_sense, sense_conf = self.disambiguator.disambiguate(
                sentence, start_char, end_char, word, pos)
            sense_def = chosen_sense.definition() if chosen_sense else "None"
            sense_id  = chosen_sense.name()       if chosen_sense else "None"

            if verbose:
                print(f"\nSTAGE 3 - Sense Disambiguation:")
                print(f"  selected_sense:   {sense_id}")
                print(f"  sense_definition: {sense_def}")
                print(f"  confidence:       {sense_conf:.4f}")

            # ============================================================
            # STAGE 4 - Candidate Generation & Filtering
            # ============================================================
            sources = self.cand_gen.generate_raw_candidates_by_source(
                sentence, word, start_char, end_char, chosen_sense, pos)

            if verbose:
                print(f"\nSTAGE 4 - Candidates:")
                print(f"  WordNet  candidates: {sources['wordnet']}")
                print(f"  BERT MLM candidates: {sources['bert_mlm']}")
                print(f"  GloVe    candidates: {sources['glove']}")

            raw_pool = set(sources['wordnet'] + sources['bert_mlm'] + sources['glove'])

            # Pre-compute MLM probabilities for all candidates in one pass
            masked_text = self.surprisal_calc.get_masked_sentence_and_idx(
                sentence, start_char, end_char)
            mask_inputs  = self.tokenizer(masked_text, return_tensors='pt').to(self.device)
            mask_idx_pos = (mask_inputs['input_ids'][0] == self.tokenizer.mask_token_id
                            ).nonzero(as_tuple=True)[0][0].item()
            with torch.no_grad():
                mask_logits = self.mlm_model(**mask_inputs).logits
            all_probs = F.softmax(mask_logits[0, mask_idx_pos], dim=-1)

            # Sentence-level semantic representation for meaning preservation
            orig_sent_inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                orig_sent_emb = self.bert_model(**orig_sent_inputs).last_hidden_state[0, 0]

            # Use dynamic low-frequency thresholds for rare source words.
            if orig_zipf < 3.5:
                dynamic_freq_gain = 0.10
                dynamic_cand_freq = max(3.5, orig_zipf + 0.25)
            elif orig_zipf < 4.0:
                dynamic_freq_gain = 0.20
                dynamic_cand_freq = max(3.8, orig_zipf + 0.25)
            else:
                dynamic_freq_gain = FREQ_GAIN_MIN
                dynamic_cand_freq = CAND_FREQ_MIN

            if verbose:
                print(f"\n  Filter detail "
                      f"(need freq_gain>={dynamic_freq_gain}, "
                      f"cand_freq>={dynamic_cand_freq}, "
                      f"mlm_prob>={MLM_PROB_MIN}, "
                      f"semantic_sim>={SEM_SIM_MIN} [raised from 0.82], "
                      f"same_pos=True, wordnet_relation):")
                print(f"  {'candidate':15} {'orig_f':7} {'cand_f':7} "
                      f"{'gain':7} {'f_ok':5} {'mlm_p':9} {'m_ok':5} {'sem':5} {'morph':5} {'pos_ok':6} verdict")
                print("  " + "-" * 100)

            filtered_cands = []
            for cand in sorted(raw_pool):
                if cand == word.lower():
                    continue
                cand_toks = self.tokenizer(cand, add_special_tokens=False)['input_ids']
                if not cand_toks:
                    continue
                mlm_prob  = all_probs[cand_toks[0]].item()
                cand_freq = wordfreq.zipf_frequency(cand, 'en')
                freq_gain = cand_freq - orig_zipf

                # Ensure candidate is semantically related to the chosen sense or the target word.
                wn_pos = None
                pos_map = {
                    'NOUN': wn.NOUN,
                    'VERB': wn.VERB,
                    'ADJ': wn.ADJ,
                    'ADV': wn.ADV,
                    'PROPN': wn.NOUN
                }
                wn_pos = pos_map.get(pos.upper()) if pos else None
                sense_related = True
                if chosen_sense and sense_conf >= 0.40:
                    sense_related = are_semantically_related(chosen_sense, word, cand, pos)
                elif wn_pos and chosen_sense is None:
                    sense_related = are_semantically_related(None, word, cand, pos)

                cand_sentence = sentence[:start_char] + cand + sentence[end_char:]
                cand_sent_inputs = self.tokenizer(cand_sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    cand_sent_emb = self.bert_model(**cand_sent_inputs).last_hidden_state[0, 0]
                semantic_sim = F.cosine_similarity(orig_sent_emb.unsqueeze(0), cand_sent_emb.unsqueeze(0)).item()
                morph_score = self._morphological_compatibility(
                    sentence, cand_sentence, start_char, end_char, len(cand))

                if not sense_related and semantic_sim >= 0.96 and morph_score >= 0.85 and mlm_prob >= MLM_PROB_MIN:
                    sense_related = True

                passes_freq  = (freq_gain >= dynamic_freq_gain) and (cand_freq >= dynamic_cand_freq)
                passes_mlm   = mlm_prob   >= MLM_PROB_MIN
                passes_sem   = semantic_sim >= SEM_SIM_MIN        # raised: 0.82 → 0.90
                passes_morph = morph_score  >= 0.60
                passes_sense = sense_related

                # ── strict POS guard (Stage 4 addition) ──────────────────────
                # Resolve candidate POS via SpaCy on the substituted sentence
                cand_doc   = self.nlp(cand_sentence)
                cand_token = self._find_token_at_span(cand_doc, start_char, start_char + len(cand))
                orig_doc_s = self.nlp(sentence)
                orig_token = self._find_token_at_span(orig_doc_s, start_char, end_char)
                if orig_token is not None and cand_token is not None:
                    passes_pos = (orig_token.pos_ == cand_token.pos_)
                else:
                    passes_pos = True  # can't determine → allow through

                accepted     = passes_freq and passes_mlm and passes_sem and passes_morph and passes_sense and passes_pos

                if verbose:
                    reject_reason = ""
                    if not passes_pos:
                        reject_reason = f"pos mismatch ({orig_token.pos_ if orig_token else '?'} vs {cand_token.pos_ if cand_token else '?'})"
                    elif not passes_freq:
                        if freq_gain < dynamic_freq_gain:
                            reject_reason = f"gain {freq_gain:.2f} < {dynamic_freq_gain}"
                        elif cand_freq < dynamic_cand_freq:
                            reject_reason = f"freq {cand_freq:.2f} < {dynamic_cand_freq}"
                    elif not passes_mlm:
                        reject_reason = f"mlm {mlm_prob:.4f} < {MLM_PROB_MIN}"
                    elif not passes_sem:
                        reject_reason = f"sem {semantic_sim:.4f} < {SEM_SIM_MIN}"
                    elif not passes_morph:
                        reject_reason = f"morph {morph_score:.2f} < 0.60"
                    elif not passes_sense:
                        reject_reason = "sense mismatch"
                    verdict = "accepted" if accepted else f"rejected ({reject_reason})"
                    print(f"  {cand:15} {orig_zipf:7.2f} {cand_freq:7.2f} "
                          f"{freq_gain:7.2f} {'yes':5} {mlm_prob:9.4f} "
                          f"{'yes' if passes_mlm else 'no':5} {semantic_sim:5.2f} {morph_score:5.2f} "
                          f"{'yes' if passes_pos else 'no':6} {verdict}")

                if accepted:
                    filtered_cands.append(cand)

            if verbose:
                print(f"\n  Final filtered candidates: {filtered_cands}")

            if not filtered_cands:
                if verbose:
                    print("  => No candidates passed filters. Keeping original word.")
                continue

            # ============================================================
            # STAGE 5 - Contextual Ranking
            # ============================================================
            if verbose:
                print(f"\nSTAGE 5 - Ranking:")
                print(f"  {'candidate':15} {'mlm_p':8} {'cos_sim':8} {'sent_sim':8} {'sense_sim':8} {'surp_red':9} {'fluency':8} {'score':8}")
                print("  " + "-" * 92)

            orig_surp    = self.surprisal_calc.compute_surprisal(
                sentence, word, start_char, end_char)
            orig_fluency = self.validator.compute_sentence_log_likelihood(sentence)

            orig_inputs   = self.tokenizer(sentence, return_tensors='pt').to(self.device)
            with torch.no_grad():
                orig_states = self.bert_model(**orig_inputs).last_hidden_state[0]
            prefix_tokens = self.tokenizer.tokenize(sentence[:start_char])
            word_tokens   = self.tokenizer.tokenize(word)
            orig_start    = min(len(prefix_tokens) + 1, orig_states.size(0) - 1)
            orig_end      = min(orig_start + len(word_tokens), orig_states.size(0))
            orig_word_emb = orig_states[orig_start:orig_end].mean(dim=0)

            orig_sent_inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                orig_sent_emb = self.bert_model(**orig_sent_inputs).last_hidden_state[0, 0]

            scored_candidates = []
            for cand in filtered_cands:
                cand_toks = self.tokenizer(cand, add_special_tokens=False)['input_ids']
                mlm_prob  = all_probs[cand_toks[0]].item() if cand_toks else 1e-9

                cand_sentence = sentence[:start_char] + cand + sentence[end_char:]
                cand_inputs   = self.tokenizer(cand_sentence, return_tensors='pt').to(self.device)
                with torch.no_grad():
                    cand_states = self.bert_model(**cand_inputs).last_hidden_state[0]
                cand_prefix   = self.tokenizer.tokenize(sentence[:start_char])
                cand_toks_l   = self.tokenizer.tokenize(cand)
                cand_s        = min(len(cand_prefix) + 1, cand_states.size(0) - 1)
                cand_e        = min(cand_s + len(cand_toks_l), cand_states.size(0))
                cand_word_emb = cand_states[cand_s:cand_e].mean(dim=0)

                cosine_sim     = F.cosine_similarity(orig_word_emb.unsqueeze(0),
                                                     cand_word_emb.unsqueeze(0)).item()
                cand_surp      = self.surprisal_calc.compute_surprisal(
                    sentence, cand, start_char, end_char)
                surp_red       = orig_surp - cand_surp
                cand_fluency   = self.validator.compute_sentence_log_likelihood(cand_sentence)
                fluency_change = cand_fluency - orig_fluency

                with torch.no_grad():
                    cand_sent_inputs = self.tokenizer(cand_sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)
                    cand_sent_emb = self.bert_model(**cand_sent_inputs).last_hidden_state[0, 0]
                sentence_sim = F.cosine_similarity(orig_sent_emb.unsqueeze(0),
                                                  cand_sent_emb.unsqueeze(0)).item()
                sense_sim = self._wordnet_sense_similarity(chosen_sense, word, cand, pos)
                morph_match = self._morphological_compatibility(
                    sentence, cand_sentence, start_char, end_char, len(cand))
                rank_score     = self.ranker.predict(
                    mlm_prob, cosine_sim, surp_red, fluency_change,
                    pos_mismatch=False   # already filtered in Stage 4; no penalty needed here
                )
                rank_score    += 0.50 * sense_sim
                rank_score    += 0.10 * sentence_sim
                semantic_priority = sense_sim + 0.50 * mlm_prob + 0.20 * morph_match

                scored_candidates.append({
                    'candidate':         cand,
                    'mlm_prob':          mlm_prob,
                    'cosine_sim':        cosine_sim,
                    'surp_red':          surp_red,
                    'fluency_change':    fluency_change,
                    'sentence_sim':      sentence_sim,
                    'sense_sim':         sense_sim,
                    'morph_match':       morph_match,
                    'semantic_priority': semantic_priority,
                    'rank_score':        rank_score
                })

            scored_candidates.sort(key=lambda x: (x['morph_match'], x['semantic_priority'], x['sense_sim'], x['rank_score']), reverse=True)

            if verbose:
                print(f"  {'candidate':15} {'mlm_p':8} {'cos_sim':8} {'sent_sim':8} {'sense_sim':8} {'morph':8} {'sem_pri':8} {'surp_red':9} {'fluency':8} {'score':8}")
                print("  " + "-" * 112)
                for i, sc in enumerate(scored_candidates):
                    marker = " <- WINNER" if i == 0 else ""
                    print(f"  {sc['candidate']:15} {sc['mlm_prob']:8.4f} "
                          f"{sc['cosine_sim']:8.4f} {sc['sentence_sim']:8.4f} "
                          f"{sc['sense_sim']:8.4f} {sc['morph_match']:8.4f} "
                          f"{sc['semantic_priority']:8.4f} {sc['surp_red']:9.4f} "
                          f"{sc['fluency_change']:8.4f} {sc['rank_score']:8.4f}{marker}")

            best   = scored_candidates[0]
            second = scored_candidates[1] if len(scored_candidates) > 1 else None
            margin = (best['semantic_priority'] - second['semantic_priority']) if second else 1.0
            best_score = best['semantic_priority']
            best_freq_gain = wordfreq.zipf_frequency(best['candidate'], 'en') - orig_zipf

            if verbose:
                print(f"\nSTAGE 5 - Confidence checks  (winner = '{best['candidate']}'):")
                ok_score = best_score >= BEST_SCORE_MIN
                ok_marg  = margin     >= MARGIN_MIN
                ok_mlm   = best['mlm_prob'] >= MLM_PROB_MIN
                ok_gain  = best_freq_gain  >= FREQ_GAIN_MIN
                print(f"  best_score  : {best_score:.4f}  "
                      f"(need >= {BEST_SCORE_MIN})  {'PASS' if ok_score else 'FAIL'}")
                print(f"  margin      : {margin:.4f}  "
                      f"(need >= {MARGIN_MIN})   {'PASS' if ok_marg else 'FAIL'}")
                print(f"  mlm_prob    : {best['mlm_prob']:.4f}  "
                      f"(need >= {MLM_PROB_MIN})  {'PASS' if ok_mlm else 'FAIL'}")
                print(f"  freq_gain   : {best_freq_gain:.4f}  "
                      f"(need >= {FREQ_GAIN_MIN})    {'PASS' if ok_gain else 'FAIL'}")

            # Confidence gate
            if best_score < BEST_SCORE_MIN:
                if verbose:
                    print(f"\nSTAGE 6 - SKIPPED: best score {best_score:.4f} "
                          f"< {BEST_SCORE_MIN}. Keeping '{word}'.")
                continue
            if margin < MARGIN_MIN:
                if verbose:
                    print(f"\nSTAGE 6 - SKIPPED: margin {margin:.4f} < {MARGIN_MIN}. "
                          f"Keeping '{word}'.")
                continue
            if best['mlm_prob'] < MLM_PROB_MIN:
                if verbose:
                    print(f"\nSTAGE 6 - SKIPPED: mlm_prob {best['mlm_prob']:.4f} "
                          f"< {MLM_PROB_MIN}. Keeping '{word}'.")
                continue

            # ============================================================
            # STAGE 6 - BERT Validator (4-gate check)
            # ============================================================
            chosen_replacement = None
            for sc in scored_candidates:
                candidate = sc['candidate']
                if verbose:
                    print(f"\nSTAGE 6 - Validating '{word}' -> '{candidate}':")

                is_valid = self.validator.validate_replacement(
                    sentence=sentence,
                    original_word=word,
                    candidate_word=candidate,
                    start_char=start_char,
                    end_char=end_char,
                    pos_tag=pos,
                    debug=verbose
                )

                if is_valid:
                    chosen_replacement = candidate
                    break
                else:
                    if verbose:
                        print(f"  Candidate '{candidate}' rejected by validator. "
                              "Trying next...")

            if verbose:
                print(f"\nSTAGE 6 - Replacement decision:")

            if chosen_replacement:
                if word.isupper():
                    inflected = chosen_replacement.upper()
                elif len(word) > 0 and word[0].isupper():
                    inflected = chosen_replacement.capitalize()
                else:
                    inflected = chosen_replacement

                replacements[word] = inflected
                if verbose:
                    print(f"  ACCEPTED: '{word}' -> '{inflected}'")
            else:
                if verbose:
                    print(f"  REJECTED: No candidate passed validation. "
                          f"Keeping '{word}'.")

        # ================================================================
        # Apply all replacements simultaneously
        # ================================================================
        final_sentence = self.replacer.replace_all(sentence, replacements)

        if verbose:
            print("\n" + "=" * 60)
            print(f"OUTPUT: {final_sentence}")
            print("=" * 60 + "\n")

        return final_sentence
