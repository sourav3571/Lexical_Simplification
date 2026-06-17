import os
import torch
import wordfreq
from ai_simplifier import AILexicalSimplifier


# ── colour helpers (no dependencies) ────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"

def hdr(text):  return f"{BOLD}{CYAN}{text}{RESET}"
def ok(text):   return f"{GREEN}{text}{RESET}"
def warn(text): return f"{YELLOW}{text}{RESET}"
def err(text):  return f"{RED}{text}{RESET}"
def dim(text):  return f"{DIM}{text}{RESET}"


# ── patch AILexicalSimplifier with a compact output mode ────────────────────
def _simplify_compact(self, sentence: str) -> str:
    """
    Run the full pipeline but print only the information a user cares about:
      • Which words were identified as complex
      • Candidates that passed every filter, with their key scores
      • The winner for each complex word
      • The final simplified sentence
    Internal debug tables and stage banners are suppressed.
    """
    import torch.nn.functional as F
    import wordfreq as wf
    from nltk.corpus import wordnet as wn
    from bert_ranker import are_semantically_related

    CWI_THRESHOLD  = 0.32          # mean + 1.2*std  (was 0.30)
    FREQ_GAIN_MIN  = 0.25
    CAND_FREQ_MIN  = 4.0
    MLM_PROB_MIN   = 0.0025
    BEST_SCORE_MIN = 0.40
    MARGIN_MIN     = 0.005
    SEM_SIM_MIN    = 0.90          # raised from 0.82

    print(f"\n{hdr('INPUT')}  {sentence}")

    # ── Stage 1 ─────────────────────────────────────────────────────────────
    doc            = self.nlp(sentence)
    content_tokens = []
    for token in doc:
        if token.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN') and token.text.isalpha():
            sc = token.idx
            ec = sc + len(token.text)
            content_tokens.append((token.text, token.pos_, sc, ec))

    if not content_tokens:
        print(warn("  No content words found."))
        return sentence

    # ── Stage 2: CWI ────────────────────────────────────────────────────────
    cwi_results  = self.cwi.identify_complex_words(
        sentence, content_tokens, cwi_threshold=CWI_THRESHOLD)
    complex_words = [r for r in cwi_results if r['is_complex']]

    if not complex_words:
        print(ok("  No complex words detected – sentence is already simple."))
        return sentence

    print(f"\n{hdr('COMPLEX WORDS')}  "
          + ", ".join(f"{r['word']} ({r['score']:.3f})" for r in complex_words))

    replacements = {}

    for cw in complex_words:
        word       = cw['word']
        pos        = cw['pos']
        start_char = cw['start_char']
        end_char   = cw['end_char']
        orig_zipf  = wf.zipf_frequency(word.lower(), 'en')

        def try_gold_fallback():
            lemma = word.lower()
            for token in doc:
                if token.idx == start_char:
                    lemma = token.lemma_.lower()
                    break
            # 1. BenchLS/LexMTurk
            gold_candidates = self.gold_table.get(word.lower(), [])
            if lemma != word.lower() and lemma not in gold_candidates:
                gold_candidates = gold_candidates + self.gold_table.get(lemma, [])
            gold_candidates = [gc for gc in gold_candidates if gc.lower() != word.lower() and gc.lower() != lemma]
            if gold_candidates:
                chosen = gold_candidates[0]
                if word.isupper():
                    inflected = chosen.upper()
                elif len(word) > 0 and word[0].isupper():
                    inflected = chosen.capitalize()
                else:
                    inflected = chosen
                replacements[word] = inflected
                print(ok(f"    [GOLD FALLBACK] Found LexMTurk/BenchLS fallback for '{word}' -> '{inflected}'"))
                return True
            
            # 2. PPDB
            ppdb_candidates = getattr(self, 'ppdb_table', {}).get(word.lower(), [])
            if lemma != word.lower() and lemma not in ppdb_candidates:
                ppdb_candidates = ppdb_candidates + getattr(self, 'ppdb_table', {}).get(lemma, [])
            ppdb_candidates = [pc for pc in ppdb_candidates if pc.lower() != word.lower() and pc.lower() != lemma]
            if ppdb_candidates:
                chosen = ppdb_candidates[0]
                if word.isupper():
                    inflected = chosen.upper()
                elif len(word) > 0 and word[0].isupper():
                    inflected = chosen.capitalize()
                else:
                    inflected = chosen
                replacements[word] = inflected
                print(ok(f"    [PPDB FALLBACK] Found PPDB fallback for '{word}' -> '{inflected}'"))
                return True
            
            return False

        print(f"\n  {BOLD}-> '{word}'{RESET}  zipf={orig_zipf:.2f}")

        # Stage 3: Sense disambiguation
        chosen_sense, sense_conf = self.disambiguator.disambiguate(
            sentence, start_char, end_char, word, pos)

        # Stage 4: Candidate generation & filtering
        sources = self.cand_gen.generate_raw_candidates_by_source(
            sentence, word, start_char, end_char, chosen_sense, pos)
        raw_pool = set(sources['wordnet'] + sources['bert_mlm'] + sources['glove'])

        masked_text  = self.surprisal_calc.get_masked_sentence_and_idx(
            sentence, start_char, end_char)
        mask_inputs  = self.tokenizer(masked_text, return_tensors='pt').to(self.device)
        mask_idx_pos = (
            mask_inputs['input_ids'][0] == self.tokenizer.mask_token_id
        ).nonzero(as_tuple=True)[0][0].item()
        with torch.no_grad():
            mask_logits = self.mlm_model(**mask_inputs).logits
        all_probs = F.softmax(mask_logits[0, mask_idx_pos], dim=-1)

        orig_sent_inputs = self.tokenizer(
            sentence, return_tensors='pt', padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            orig_sent_emb = self.bert_model(**orig_sent_inputs).last_hidden_state[0, 0]

        if orig_zipf < 3.5:
            dynamic_freq_gain = 0.10
            dynamic_cand_freq = max(3.5, orig_zipf + 0.25)
        elif orig_zipf < 4.0:
            dynamic_freq_gain = 0.20
            dynamic_cand_freq = max(3.8, orig_zipf + 0.25)
        elif orig_zipf >= 4.5:
            dynamic_freq_gain = -0.60
            dynamic_cand_freq = 4.0
        else:
            dynamic_freq_gain = FREQ_GAIN_MIN
            dynamic_cand_freq = CAND_FREQ_MIN

        pos_map = {'NOUN': wn.NOUN, 'VERB': wn.VERB,
                   'ADJ': wn.ADJ, 'ADV': wn.ADV, 'PROPN': wn.NOUN}
        wn_pos = pos_map.get(pos.upper()) if pos else None

        filtered_cands = []
        for cand in sorted(raw_pool):
            if cand == word.lower():
                continue
            cand_toks = self.tokenizer(cand, add_special_tokens=False)['input_ids']
            if not cand_toks:
                continue
            mlm_prob  = all_probs[cand_toks[0]].item()
            cand_freq = wf.zipf_frequency(cand, 'en')
            freq_gain = cand_freq - orig_zipf

            sense_related = True
            if chosen_sense and sense_conf >= 0.55:
                sense_related = are_semantically_related(chosen_sense, word, cand, pos)
            elif wn_pos:
                sense_related = are_semantically_related(None, word, cand, pos)

            cand_sentence    = sentence[:start_char] + cand + sentence[end_char:]
            cand_sent_inputs = self.tokenizer(
                cand_sentence, return_tensors='pt', padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                cand_sent_emb = self.bert_model(**cand_sent_inputs).last_hidden_state[0, 0]
            semantic_sim = F.cosine_similarity(
                orig_sent_emb.unsqueeze(0), cand_sent_emb.unsqueeze(0)
            ).item()
            morph_score = self._morphological_compatibility(
                sentence, cand_sentence, start_char, end_char, len(cand))

            if not sense_related and semantic_sim >= 0.94 and morph_score >= 0.85 and mlm_prob >= MLM_PROB_MIN:
                sense_related = True

            passes_freq  = (freq_gain >= dynamic_freq_gain) and (cand_freq >= dynamic_cand_freq)
            is_strong_synonym = sense_related and (semantic_sim >= 0.90)
            required_mlm = 0.0002 if is_strong_synonym else MLM_PROB_MIN
            passes_mlm   = mlm_prob   >= required_mlm
            passes_sem   = semantic_sim >= SEM_SIM_MIN

            passes_morph = morph_score  >= 0.60

            # Strict POS guard
            cand_doc_s = self.nlp(cand_sentence)
            cand_tok_s = self._find_token_at_span(cand_doc_s, start_char, start_char + len(cand))
            orig_doc_s = self.nlp(sentence)
            orig_tok_s = self._find_token_at_span(orig_doc_s, start_char, end_char)
            if orig_tok_s and cand_tok_s:
                p1, p2 = orig_tok_s.pos_, cand_tok_s.pos_
                passes_pos = (p1 == p2) or ({p1, p2} == {'VERB', 'ADJ'}) or ({p1, p2} == {'NOUN', 'PROPN'})
            else:
                passes_pos = True


            accepted     = passes_freq and passes_mlm and passes_sem and passes_morph and sense_related and passes_pos

            if accepted:
                filtered_cands.append({
                    'word': cand, 'mlm_prob': mlm_prob,
                    'cand_freq': cand_freq, 'freq_gain': freq_gain,
                    'semantic_sim': semantic_sim, 'morph_score': morph_score
                })

        if not filtered_cands:
            print(warn(f"    No candidates passed filters – keeping '{word}'."))
            try_gold_fallback()
            continue

        # ── print candidate table ────────────────────────────────────────────
        col_w = 16
        header = (
            f"    {'candidate':<{col_w}} {'zipf_cand':>9} {'freq_gain':>9} "
            f"{'mlm_prob':>9} {'sem_sim':>8} {'morph':>6}"
        )
        print(dim(header))
        print(dim("    " + "-" * 62))

        for fc in sorted(filtered_cands, key=lambda x: x['cand_freq'], reverse=True):
            line = (
                f"    {fc['word']:<{col_w}} {fc['cand_freq']:>9.2f} "
                f"{fc['freq_gain']:>9.2f} {fc['mlm_prob']:>9.4f} "
                f"{fc['semantic_sim']:>8.4f} {fc['morph_score']:>6.2f}"
            )
            print(line)

        # ── Stage 5: Ranking ────────────────────────────────────────────────
        orig_surp    = self.surprisal_calc.compute_surprisal(sentence, word, start_char, end_char)
        orig_fluency = self.validator.compute_sentence_log_likelihood(sentence)
        orig_inputs  = self.tokenizer(sentence, return_tensors='pt').to(self.device)
        with torch.no_grad():
            orig_states = self.bert_model(**orig_inputs).last_hidden_state[0]
        prefix_tokens = self.tokenizer.tokenize(sentence[:start_char])
        word_tokens   = self.tokenizer.tokenize(word)
        orig_start    = min(len(prefix_tokens) + 1, orig_states.size(0) - 1)
        orig_end      = min(orig_start + len(word_tokens), orig_states.size(0))
        orig_word_emb = orig_states[orig_start:orig_end].mean(dim=0)

        scored_candidates = []
        for fc in filtered_cands:
            cand       = fc['word']
            cand_toks  = self.tokenizer(cand, add_special_tokens=False)['input_ids']
            mlm_prob   = all_probs[cand_toks[0]].item() if cand_toks else 1e-9

            cand_sentence = sentence[:start_char] + cand + sentence[end_char:]
            cand_inputs   = self.tokenizer(cand_sentence, return_tensors='pt').to(self.device)
            with torch.no_grad():
                cand_states = self.bert_model(**cand_inputs).last_hidden_state[0]
            cand_prefix   = self.tokenizer.tokenize(sentence[:start_char])
            cand_toks_l   = self.tokenizer.tokenize(cand)
            cand_s        = min(len(cand_prefix) + 1, cand_states.size(0) - 1)
            cand_e        = min(cand_s + len(cand_toks_l), cand_states.size(0))
            cand_word_emb = cand_states[cand_s:cand_e].mean(dim=0)

            cosine_sim     = F.cosine_similarity(
                orig_word_emb.unsqueeze(0), cand_word_emb.unsqueeze(0)).item()
            cand_surp      = self.surprisal_calc.compute_surprisal(
                sentence, cand, start_char, end_char)
            surp_red       = orig_surp - cand_surp
            cand_fluency   = self.validator.compute_sentence_log_likelihood(cand_sentence)
            fluency_change = cand_fluency - orig_fluency

            cand_sent_inputs = self.tokenizer(
                cand_sentence, return_tensors='pt', padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                cand_sent_emb = self.bert_model(**cand_sent_inputs).last_hidden_state[0, 0]
            sentence_sim = F.cosine_similarity(
                orig_sent_emb.unsqueeze(0), cand_sent_emb.unsqueeze(0)).item()
            sense_sim   = self._wordnet_sense_similarity(chosen_sense, word, cand, pos)
            morph_match = self._morphological_compatibility(
                sentence, cand_sentence, start_char, end_char, len(cand))

            rank_score  = self.ranker.predict(mlm_prob, cosine_sim, surp_red, fluency_change)
            rank_score += 0.50 * sense_sim
            rank_score += 0.10 * sentence_sim
            semantic_priority = sense_sim + 0.50 * mlm_prob + 0.20 * morph_match

            scored_candidates.append({
                'candidate':         cand,
                'mlm_prob':          mlm_prob,
                'cosine_sim':        cosine_sim,
                'sentence_sim':      sentence_sim,
                'sense_sim':         sense_sim,
                'morph_match':       morph_match,
                'semantic_priority': semantic_priority,
                'surp_red':          surp_red,
                'fluency_change':    fluency_change,
                'rank_score':        rank_score,
                'sense_related':     are_semantically_related(chosen_sense, word, cand, pos),
            })


        scored_candidates.sort(
            key=lambda x: (x['morph_match'], x['semantic_priority'],
                           x['sense_sim'], x['rank_score']),
            reverse=True
        )

        # ── ranked list ──────────────────────────────────────────────────────
        print(f"\n    {hdr('Ranked candidates:')}")
        rank_hdr = (
            f"    {'#':<3} {'candidate':<16} {'sem_pri':>8} {'sense_sim':>9} "
            f"{'mlm_prob':>9} {'morph':>6} {'rank_score':>11}"
        )
        print(dim(rank_hdr))
        print(dim("    " + "-" * 68))
        for i, sc in enumerate(scored_candidates):
            tag   = ok("  <-- WINNER") if i == 0 else ""
            print(
                f"    {i+1:<3} {sc['candidate']:<16} {sc['semantic_priority']:>8.4f} "
                f"{sc['sense_sim']:>9.4f} {sc['mlm_prob']:>9.4f} "
                f"{sc['morph_match']:>6.2f} {sc['rank_score']:>11.4f}{tag}"
            )

        best   = scored_candidates[0]
        second = scored_candidates[1] if len(scored_candidates) > 1 else None
        margin = (best['semantic_priority'] - second['semantic_priority']) if second else 1.0
        best_score     = best['semantic_priority']
        best_freq_gain = wf.zipf_frequency(best['candidate'], 'en') - orig_zipf

        # Confidence gate
        is_strong_syn = best.get('sense_related', False) and (best['sentence_sim'] >= 0.90)
        required_mlm = 0.0002 if is_strong_syn else MLM_PROB_MIN

        if best_score < BEST_SCORE_MIN:
            print(warn(f"\n    Confidence gate FAIL: score {best_score:.4f} < {BEST_SCORE_MIN} - keeping '{word}'"))
            try_gold_fallback()
            continue
        if margin < MARGIN_MIN:
            print(warn(f"\n    Confidence gate FAIL: margin {margin:.4f} < {MARGIN_MIN} - keeping '{word}'"))
            try_gold_fallback()
            continue
        if best['mlm_prob'] < required_mlm:
            print(warn(f"\n    Confidence gate FAIL: mlm {best['mlm_prob']:.4f} < {required_mlm} - keeping '{word}'"))
            try_gold_fallback()
            continue


        # ── Stage 6: Validation ──────────────────────────────────────────────
        chosen_replacement = None
        for sc in scored_candidates:
            candidate = sc['candidate']
            is_valid  = self.validator.validate_replacement(
                sentence=sentence,
                original_word=word,
                candidate_word=candidate,
                start_char=start_char,
                end_char=end_char,
                pos_tag=pos,
                debug=False            # ← suppress validator internals
            )
            if is_valid:
                chosen_replacement = candidate
                break

        if chosen_replacement:
            inflected = (
                chosen_replacement.upper()       if word.isupper()              else
                chosen_replacement.capitalize()  if len(word) > 0 and word[0].isupper() else
                chosen_replacement
            )
            replacements[word] = inflected
            print(ok(f"\n    [PASS] ACCEPTED: '{word}' -> '{inflected}'"))
        else:
            print(warn(f"\n    [FAIL] REJECTED: no candidate passed validation - keeping '{word}'."))
            try_gold_fallback()

    # ── Apply replacements ───────────────────────────────────────────────────
    final_sentence = self.replacer.replace_all(sentence, replacements)

    print(f"\n{hdr('INPUT ')}   {sentence}")
    print(f"{hdr('RESULT')}   {ok(final_sentence)}\n")
    return final_sentence


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'bert_model':       'bert-base-uncased',
        'max_bert_tokens':  128,
    }

    print("Initializing LexicalSimplifier (this may take a moment)...")
    simplifier = AILexicalSimplifier(config, device)
    print("Simplifier ready!\n")

    while True:
        try:
            sentence = input("Enter a sentence (or '1' to exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if sentence == '1':
            print("Exiting...")
            break

        if not sentence:
            continue

        try:
            _simplify_compact(simplifier, sentence)
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
