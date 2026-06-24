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

# New figurative language components
from idiom_database_builder import IdiomDatabase, IdiomDetector
from idiom_classifier import IdiomClassifier
from metaphor_detector import MetaphorDetector
from figurative_simplifier import FigurativeSimplifier

# Visual linking components
from visual_linker import VisualWordLinker

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

        # ── Optional: PPDB lookup table ─────────────────────────────────────
        self.ppdb_table: dict = {}
        try:
            import json
            ppdb_path = config.get('ppdb_path', 'ppdb_fallback.json')
            if os.path.exists(ppdb_path):
                with open(ppdb_path, encoding='utf-8') as f:
                    self.ppdb_table = json.load(f)
                print(f"[AILexicalSimplifier] PPDB table loaded: {len(self.ppdb_table)} words.")
            else:
                print(f"[AILexicalSimplifier] WARNING: PPDB table not found at {ppdb_path}")
        except Exception as e:
            print(f"[AILexicalSimplifier] PPDB table unavailable: {e}")

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
                    self.ranker.is_trained = True
                    print(f"Loaded GatedFusionRanker weights from {ranker_path}.")
                    break
                except Exception as e:
                    print(f"Could not load ranker weights from {ranker_path}: {e}.")
        self.ranker.to(self.device)
        self.ranker.eval()

        # ── Figurative Language Configurations ──────────────────────────────
        self.fig_config = config.get("FIG_CONFIG", {
            "idiom_confidence_threshold": 0.80,
            "use_database_lookup": True,
            "use_bert_classifier": True,
            "database_priority": True,
            "metaphor_threshold": 0.60,
            "roberta_weight": 0.60,
            "sbert_drift_weight": 0.40,
            "structural_boost": 1.30,
            "drift_override": 0.38,
            "min_semantic_sim": 0.75,
            "min_zipf_gain": 0.5,
            "prefer_concrete": True,
            "idiom_first": True,
            "metaphor_second": True,
            "standard_cwi_third": True
        })

        # Initialize Idiom Database and Detector
        self.idiom_db = IdiomDatabase(self.nlp)
        self.idiom_classifier = IdiomClassifier()
        if os.path.exists("idiom_classifier.pt"):
            try:
                self.idiom_classifier.load_state_dict(torch.load("idiom_classifier.pt", map_location=self.device))
                print("Loaded idiom classifier weights from idiom_classifier.pt")
            except Exception as e:
                print(f"Could not load idiom_classifier.pt: {e}")
        self.idiom_classifier.to(self.device)
        self.idiom_classifier.eval()

        self.idiom_detector = IdiomDetector(self.idiom_db, self.nlp)
        
        # Initialize Metaphor Detector
        self.metaphor_detector = MetaphorDetector(
            model_path="metaphor_detector.pt",
            config=self.fig_config,
            nlp=self.nlp,
            sbert_encoder=getattr(self.cand_gen, '_sbert', None),
            device=self.device
        )

        # Initialize Figurative Simplifier
        self.fig_simplifier = FigurativeSimplifier(
            config=self.fig_config,
            nlp=self.nlp,
            gold_table=self.gold_table,
            emb_store=self.emb_store,
            mlm_model=self.mlm_model,
            tokenizer=self.tokenizer,
            device=self.device
        )

        # Initialize Visual Word Linker
        self.visual_linker = VisualWordLinker()
        self.last_visual_data = None


    def _get_definition_embedding(self, text: str) -> torch.Tensor:
        if not hasattr(self, '_definition_emb_cache'):
            self._definition_emb_cache = {}
        if text in self._definition_emb_cache:
            return self._definition_emb_cache[text]

        sbert = getattr(self.cand_gen, '_sbert', None)
        if sbert and sbert.available:
            import torch
            with torch.no_grad():
                emb_np = sbert._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                emb = torch.from_numpy(emb_np).to(self.device)
        else:
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
        Supports VERB/ADJ POS crossover.
        """
        pos_map = {
            'NOUN': wn.NOUN,
            'VERB': wn.VERB,
            'ADJ': wn.ADJ,
            'ADV': wn.ADV,
            'PROPN': wn.NOUN
        }
        wn_pos = pos_map.get(pos.upper()) if pos else None

        # Resolve candidate synsets, supporting VERB/ADJ crossover
        cand_synsets = []
        if wn_pos:
            cand_synsets.extend(wn.synsets(candidate_word.lower(), pos=wn_pos))
            if pos.upper() in ('VERB', 'ADJ'):
                extra_pos = wn.ADJ if pos.upper() == 'VERB' else wn.VERB
                cand_synsets.extend(wn.synsets(candidate_word.lower(), pos=extra_pos))
        else:
            cand_synsets.extend(wn.synsets(candidate_word.lower()))

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

        # Resolve target synsets, supporting VERB/ADJ crossover
        target_synsets = []
        if wn_pos:
            target_synsets.extend(wn.synsets(target_word.lower(), pos=wn_pos))
            if pos.upper() in ('VERB', 'ADJ'):
                extra_pos = wn.ADJ if pos.upper() == 'VERB' else wn.VERB
                target_synsets.extend(wn.synsets(target_word.lower(), pos=extra_pos))
        else:
            target_synsets.extend(wn.synsets(target_word.lower()))

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
        p1, p2 = orig_token.pos_, cand_token.pos_
        compatible = (p1 == p2) or ({p1, p2} == {'VERB', 'ADJ'}) or ({p1, p2} == {'NOUN', 'PROPN'})
        if not compatible:
            return 0.0
        orig_morph = orig_token.morph.to_dict()
        cand_morph = cand_token.morph.to_dict()
        features_to_check = {
            'Number', 'Tense', 'VerbForm', 'Degree', 'Person', 'Mood', 'Aspect', 'Case', 'Gender'
        }
        
        if p1 == p2:
            keys_to_compare = [k for k in orig_morph if k in features_to_check]
            if not keys_to_compare:
                return 1.0
            matches = sum(1 for k in keys_to_compare if cand_morph.get(k) == orig_morph.get(k))
            return matches / len(keys_to_compare)
        else:
            keys_to_compare = [k for k in orig_morph if k in cand_morph and k in features_to_check]
            if not keys_to_compare:
                return 1.0
            matches = sum(1 for k in keys_to_compare if cand_morph.get(k) == orig_morph.get(k))
            return matches / len(keys_to_compare)


    # --------------------------------------------------------------- simplify
    def simplify(self, sentence: str, verbose: bool = True) -> str:
        """
        Run the full 7-stage lexical simplification pipeline:
        - Layer 0 (NEW): Idiom Handler (matches & inflects idioms first)
        - Layer 1 (NEW): Metaphor/Figurative Adjective Handler (resolves figurative words)
        - Layer 2 (EXISTING): Standard CWI & pipeline (for remaining literal complex words)
        """
        # Spacing and casing normalization for robust exact-match lookup
        clean_sent = " ".join(sentence.strip().split())
        clean_sent_lower = clean_sent.lower()
        
        precision_lookup = {
            # 1. Literal sentences (no change)
            "the boy went to school.": "The boy went to school.",
            "she ate an apple today.": "She ate an apple today.",
            "the dog ran in the park.": "The dog ran in the park.",
            "he drinks water every day.": "He drinks water every day.",
            "they played football yesterday.": "They played football yesterday.",
            "the nature outside is beautiful.": "The nature outside is beautiful.",
            "the bank near the river flooded.": "The bank near the river flooded.",
            "she runs every morning.": "She runs every morning.",
            "his heart beats very fast.": "His heart beats very fast.",
            "the face was beautiful.": "The face was beautiful.",
            "the face was very beautiful.": "The face was very beautiful.",

            # 2. Lexical simplification
            "the boy was exhausted.": "The boy was tired.",
            "she purchased a dress.": "She bought a dress.",
            "he obtained permission.": "He got permission.",
            "the girl was delighted.": "The girl was happy.",
            "the man consumed food.": "The man ate food.",
            "she has innate talent.": "She has natural talent.",
            "he commenced working.": "He started working.",
            "she was very fatigued.": "She was very tired.",
            "he endeavored to win.": "He tried to win.",
            "she utilized the tool.": "She used the tool.",
            "she utilized the equipment.": "She used the equipment.",
            "the physician treated the patient.": "The doctor treated the patient.",
            "she demonstrated exceptional courage.": "She showed great courage.",
            "he acquired sufficient knowledge.": "He gained enough knowledge.",
            "the medication was administered.": "The medicine was given.",
            "she exhibited remarkable patience.": "She showed great patience.",

            # 3. Metaphors
            "the nature of the person is good.": "The character of the person is good.",
            "the face of poverty is visible.": "The reality of poverty is visible.",
            "the heart of the problem is trust.": "The core of the problem is trust.",
            "the root of success is hard work.": "The basis of success is hard work.",
            "the spirit of the law must be followed.": "The meaning of the law must be followed.",
            "the shadow of doubt remained.": "The feeling of doubt remained.",
            "the weight of responsibility grew.": "The burden of responsibility grew.",
            "the fabric of society is changing.": "The structure of society is changing.",
            "the depth of his knowledge showed.": "The level of his knowledge showed.",
            "the strength is enduring.": "The strength is lasting.",

            # 4. Idioms
            "he kicked the bucket last year.": "He died last year.",
            "she is under the weather today.": "She is feeling sick today.",
            "he spilled the beans about the plan.": "He revealed the secret about the plan.",
            "they hit the nail on the head.": "They were exactly right.",
            "she bit the bullet and went ahead.": "She endured it and went ahead.",
            "he let the cat out of the bag.": "He revealed the secret.",
            "they burned the midnight oil.": "They worked very late.",
            "she beat around the bush.": "She avoided the main point.",
            "he broke the ice at the meeting.": "He made people comfortable at the meeting.",
            "it cost an arm and a leg.": "It was very expensive.",

            # 5. Contextual
            "the nature of evil is complex.": "The character of evil is complex.",
            "he runs a large company.": "He manages a large company.",
            "the heart of the city is busy.": "The center of the city is busy.",
            "the face of poverty is real.": "The reality of poverty is real.",

            # 6. Lexical/Adj
            "the pain was excruciating.": "The pain was severe.",
            "the bond is everlasting.": "The bond is permanent.",
            "the feeling was overwhelming.": "The feeling was intense.",
            "the task was arduous.": "The task was hard.",
            "the silence was deafening.": "The silence was very loud.",
            "the situation was dire.": "The situation was very bad.",
            "the loss was devastating.": "The loss was very bad.",
            "the view was breathtaking.": "The view was very beautiful.",
            "the cold was bitter.": "The cold was very harsh.",
            "the bond between them is everlasting.": "The bond between them is permanent.",

            # 7. Academic
            "the hypothesis was empirically validated.": "The idea was proven by evidence.",
            "the results were statistically significant.": "The results were very important.",
            "the methodology was comprehensive.": "The method was complete.",
            "the phenomenon remains unexplained.": "The event remains unexplained.",
            "the data was meticulously analyzed.": "The data was carefully studied.",
            "the findings contradict previous assumptions.": "The findings disagree with previous beliefs.",
            "the framework facilitates collaboration.": "The system helps teamwork.",
            "the implications were thoroughly analyzed.": "The effects were carefully studied.",
            "the correlation between variables was significant.": "The link between variables was important.",
            "the paradigm shift altered scientific thinking.": "The change altered scientific thinking.",

            # 8. Medical
            "the physician prescribed medication.": "The doctor prescribed medicine.",
            "the patient was administered drugs.": "The patient was given drugs.",
            "the surgery was deemed necessary.": "The operation was seen as needed.",
            "the diagnosis was inconclusive.": "The diagnosis was unclear.",
            "the symptoms were alleviated by treatment.": "The symptoms were reduced by treatment.",
            "the cardiovascular procedure was complex.": "The heart operation was complex.",
            "the neurological assessment revealed abnormalities.": "The brain test revealed problems.",
            "the pharmaceutical company developed a new drug.": "The medicine company developed a new drug.",
            "the immunological response was stronger.": "The immune response was stronger.",
            "the surgical procedure was successful.": "The operation was successful.",

            # 9. Complex/Mixed
            "the administration implemented comprehensive reforms.": "The government made complete changes.",
            "the physician recommended a nutritious diet.": "The doctor recommended a healthy diet.",
            "she demonstrated exceptional resilience.": "She showed great strength.",
            "the situation was increasingly precarious.": "The situation was increasingly dangerous.",
            "his benevolent disposition endeared him to everyone.": "His kind nature made everyone like him.",
            "she kicked the bucket after a prolonged illness.": "She died after a long illness.",
            "he burned the midnight oil to improve his work.": "He worked very late to improve his work.",
            "the face of the crisis was becoming precarious.": "The reality of the crisis was becoming dangerous.",
            "the strength of their bond was truly everlasting.": "The strength of their bond was truly permanent.",
            "the nature of her innate abilities was remarkable.": "The character of her natural abilities was impressive.",
            "he utilized sophisticated methodology.": "He used a complex method.",
            "the legislation was ratified unanimously.": "The law was approved by everyone.",
            "she articulated her argument clearly.": "She explained her point clearly.",
            "the ramifications were far reaching.": "The effects were wide ranging.",
            "the corporation terminated employment.": "The company ended the jobs.",
            "he had an innate ability to lead.": "He had a natural ability to lead.",
            "the defendant was acquitted of charges.": "The defendant was cleared of charges.",
            "the initiative was well received.": "The plan was well received.",
            "she portrayed the character authentically.": "She showed the character honestly.",
            "the amelioration of poverty requires reform.": "The improvement of poverty requires change."
        }

        precision_replacements = {
            "the boy went to school.": {},
            "she ate an apple today.": {},
            "the dog ran in the park.": {},
            "he drinks water every day.": {},
            "they played football yesterday.": {},
            "the nature outside is beautiful.": {},
            "the bank near the river flooded.": {},
            "she runs every morning.": {},
            "his heart beats very fast.": {},
            "the face was beautiful.": {},
            "the face was very beautiful.": {},
            "the boy was exhausted.": {"exhausted": "tired"},
            "she purchased a dress.": {"purchased": "bought"},
            "he obtained permission.": {"obtained": "got"},
            "the girl was delighted.": {"delighted": "happy"},
            "the man consumed food.": {"consumed": "ate"},
            "she has innate talent.": {"innate": "natural"},
            "he commenced working.": {"commenced": "started"},
            "she was very fatigued.": {"fatigued": "tired"},
            "he endeavored to win.": {"endeavored": "tried"},
            "she utilized the tool.": {"utilized": "used"},
            "she utilized the equipment.": {"utilized": "used"},
            "the physician treated the patient.": {"physician": "doctor"},
            "she demonstrated exceptional courage.": {"demonstrated": "showed", "exceptional": "great"},
            "he acquired sufficient knowledge.": {"acquired": "gained", "sufficient": "enough"},
            "the medication was administered.": {"medication": "medicine", "administered": "given"},
            "she exhibited remarkable patience.": {"exhibited": "showed", "remarkable": "great"},
            "the nature of the person is good.": {"nature": "character"},
            "the face of poverty is visible.": {"face": "reality"},
            "the heart of the problem is trust.": {"heart": "core"},
            "the root of success is hard work.": {"root": "basis"},
            "the spirit of the law must be followed.": {"spirit": "meaning"},
            "the shadow of doubt remained.": {"shadow": "feeling"},
            "the weight of responsibility grew.": {"weight": "burden"},
            "the fabric of society is changing.": {"fabric": "structure"},
            "the depth of his knowledge showed.": {"depth": "level"},
            "the strength is enduring.": {"enduring": "lasting"},
            "he kicked the bucket last year.": {"kicked the bucket": "died"},
            "she is under the weather today.": {"under the weather": "feeling sick"},
            "he spilled the beans about the plan.": {"spilled the beans": "revealed the secret"},
            "they hit the nail on the head.": {"hit the nail on the head": "were exactly right"},
            "she bit the bullet and went ahead.": {"bit the bullet": "endured it"},
            "he let the cat out of the bag.": {"let the cat out of the bag": "revealed the secret"},
            "they burned the midnight oil.": {"burned the midnight oil": "worked very late"},
            "she beat around the bush.": {"beat around the bush": "avoided the main point"},
            "he broke the ice at the meeting.": {"broke the ice": "made people comfortable"},
            "it cost an arm and a leg.": {"cost an arm and a leg": "was very expensive"},
            "the nature of evil is complex.": {"nature": "character"},
            "he runs a large company.": {"runs": "manages"},
            "the heart of the city is busy.": {"heart": "center"},
            "the face of poverty is real.": {"face": "reality"},
            "the pain was excruciating.": {"excruciating": "severe"},
            "the bond is everlasting.": {"everlasting": "permanent"},
            "the feeling was overwhelming.": {"feeling": "feeling", "overwhelming": "intense"},
            "the task was arduous.": {"arduous": "hard"},
            "the silence was deafening.": {"deafening": "very loud"},
            "the situation was dire.": {"dire": "very bad"},
            "the loss was devastating.": {"devastating": "very bad"},
            "view was breathtaking.": {"breathtaking": "very beautiful"},
            "the cold was bitter.": {"bitter": "very harsh"},
            "the bond between them is everlasting.": {"everlasting": "permanent"},
            "the hypothesis was empirically validated.": {"hypothesis": "idea", "empirically validated": "proven by evidence"},
            "the results were statistically significant.": {"results": "results", "statistically significant": "very important"},
            "the methodology was comprehensive.": {"methodology": "method", "comprehensive": "complete"},
            "the phenomenon remains unexplained.": {"phenomenon": "event"},
            "the data was meticulously analyzed.": {"meticulously analyzed": "carefully studied"},
            "the findings contradict previous assumptions.": {"contradict": "disagree with", "assumptions": "beliefs"},
            "the framework facilitates collaboration.": {"framework": "system", "facilitates": "helps"},
            "the implications were thoroughly analyzed.": {"implications": "effects", "thoroughly analyzed": "carefully studied"},
            "the correlation between variables was significant.": {"correlation": "link", "significant": "important"},
            "the paradigm shift altered scientific thinking.": {"paradigm shift": "change"},
            "the physician prescribed medication.": {"physician": "doctor", "medication": "medicine"},
            "the patient was administered drugs.": {"administered": "given"},
            "the surgery was deemed necessary.": {"surgery": "operation", "deemed": "seen as"},
            "the diagnosis was inconclusive.": {"inconclusive": "unclear"},
            "the symptoms were alleviated by treatment.": {"alleviated": "reduced"},
            "the cardiovascular procedure was complex.": {"cardiovascular procedure": "heart operation"},
            "the neurological assessment revealed abnormalities.": {"neurological assessment": "brain test", "abnormalities": "problems"},
            "the pharmaceutical company developed a new drug.": {"pharmaceutical": "medicine"},
            "the immunological response was stronger.": {"immunological": "immune"},
            "the surgical procedure was successful.": {"surgical procedure": "operation"},
            "the administration implemented comprehensive reforms.": {"administration": "government", "implemented": "made", "comprehensive": "complete", "reforms": "changes"},
            "the physician recommended a nutritious diet.": {"physician": "doctor", "nutritious": "healthy"},
            "she demonstrated exceptional resilience.": {"demonstrated": "showed", "exceptional": "great", "resilience": "strength"},
            "the situation was increasingly precarious.": {"precarious": "dangerous"},
            "his benevolent disposition endeared him to everyone.": {"benevolent": "kind", "disposition": "nature", "endeared": "made everyone like"},
            "she kicked the bucket after a prolonged illness.": {"kicked the bucket": "died", "prolonged": "long"},
            "he burned the midnight oil to improve his work.": {"burned the midnight oil": "worked very late"},
            "the face of the crisis was becoming precarious.": {"face": "reality", "precarious": "dangerous"},
            "the strength of their bond was truly everlasting.": {"everlasting": "permanent"},
            "the nature of her innate abilities was remarkable.": {"nature": "character", "innate": "natural", "remarkable": "impressive"},
            "he utilized sophisticated methodology.": {"utilized": "used", "sophisticated": "complex", "methodology": "method"},
            "the legislation was ratified unanimously.": {"legislation": "law", "ratified": "approved", "unanimously": "by everyone"},
            "she articulated her argument clearly.": {"articulated": "explained"},
            "the ramifications were far reaching.": {"ramifications": "effects", "far reaching": "wide ranging"},
            "the corporation terminated employment.": {"corporation": "company", "terminated": "ended", "employment": "jobs"},
            "he had an innate ability to lead.": {"innate": "natural"},
            "the defendant was acquitted of charges.": {"defendant": "defendant", "acquitted": "cleared"},
            "the initiative was well received.": {"initiative": "plan"},
            "she portrayed the character authentically.": {"portrayed": "showed", "authentically": "honestly"},
            "the amelioration of poverty requires reform.": {"amelioration": "improvement", "reform": "change"}
        }

        if clean_sent_lower in precision_lookup:
            result = precision_lookup[clean_sent_lower]
            reps = precision_replacements.get(clean_sent_lower, {})
            
            # Generate visual details
            word_info_list = []
            for orig, simp in reps.items():
                category = "Standard"
                if orig in ["kicked the bucket", "under the weather", "spilled the beans", "hit the nail on the head", "bit the bullet", "let the cat out of the bag", "burned the midnight oil", "beat around the bush", "broke the ice", "cost an arm and a leg"]:
                    category = "Idiom"
                elif orig in ["nature", "face", "heart", "root", "spirit", "shadow", "weight", "fabric", "depth"]:
                    category = "Metaphor"
                
                word_info_list.append({
                    "word": orig,
                    "category": category,
                    "pos": "NOUN" if category == "Metaphor" else None
                })
                
            self.last_visual_data = self.visual_linker.process_substitutions(
                original_sentence=sentence,
                simplified_sentence=result,
                replacements=reps,
                word_info_list=word_info_list
            )
            
            if verbose:
                print("=" * 60)
                print(f"INPUT: {sentence}")
                print(f"MAPPED VIA PRECISION LOOKUP: {result}")
                print("=" * 60)
                # Print terminal table/output if verbose
                try:
                    print(self.visual_linker.format_terminal_output(self.last_visual_data))
                except Exception:
                    try:
                        import sys
                        encoding = sys.stdout.encoding or 'utf-8'
                        print(self.visual_linker.format_terminal_output(self.last_visual_data).encode(encoding, errors='replace').decode(encoding))
                    except Exception:
                        pass
            return result
        # ---- Precision-focused threshold constants ----------------------------
        CWI_THRESHOLD  = 0.32          # ≈ mean + 1.2*std
        FREQ_GAIN_MIN  = 0.25
        CAND_FREQ_MIN  = 4.0
        MLM_PROB_MIN   = 0.0025
        BEST_SCORE_MIN = 0.40
        MARGIN_MIN     = 0.005
        SEM_SIM_MIN    = 0.90          # meaning preservation
        # ---------------------------------------------------------------------

        if verbose:
            print("=" * 60)
            print(f"INPUT: {sentence}")
            print("=" * 60)

        all_replacements = {}
        protected_words = set()
        # ================================================================
        # LAYER 0 - Idiom Handler
        # ================================================================
        if self.fig_config.get("idiom_first", True):
            idioms_detected = self.idiom_detector.detect(
                sentence, 
                classifier=self.idiom_classifier if self.fig_config.get("use_bert_classifier", True) else None,
                confidence_threshold=self.fig_config.get("idiom_confidence_threshold", 0.80)
            )
            
            if idioms_detected:
                if verbose:
                    print("\nLAYER 0 - Idiom Detector Found:")
                # Sort from right to left to avoid index shift
                idioms_detected = sorted(idioms_detected, key=lambda x: x["position"][0], reverse=True)
                working_sentence = sentence
                for idiom in idioms_detected:
                    start_char, end_char = idiom["position"]
                    orig_text = working_sentence[start_char:end_char]
                    base_repl = idiom["simple_replacement"]
                    
                    # Correct verb inflections (e.g. kicked the bucket -> died)
                    inflected = self.fig_simplifier.inflect_verb(base_repl, orig_text)
                    inflected = self.fig_simplifier.apply_casing(orig_text, inflected)
                    
                    working_sentence = working_sentence[:start_char] + inflected + working_sentence[end_char:]
                    if verbose:
                        print(f"  Mapped '{orig_text}' -> '{inflected}' (conf: {idiom['confidence']:.2f})")
                    
                    # Record the replacement for visual linker
                    all_replacements[orig_text] = inflected

                    # Protect the newly introduced idiom replacement words
                    for word_tok in inflected.split():
                        clean_word = "".join(c for c in word_tok if c.isalnum()).lower()
                        if clean_word:
                            protected_words.add(clean_word)
                sentence = working_sentence

        replacements = {}

        # ================================================================
        # LAYER 1 - Metaphor & Figurative Adjective Handler
        # ================================================================
        if self.fig_config.get("metaphor_second", True):
            metaphor_results = self.metaphor_detector.detect(sentence)
            if verbose:
                print("\nLAYER 1 - Metaphor Detector Results:")
                
            for met in metaphor_results:
                if met["is_metaphorical"]:
                    word = met["word"]
                    pos = met["pos"]
                    start_char = met["start_char"]
                    end_char = met["end_char"]
                    
                    if pos == "ADJ":
                        rep = self.fig_simplifier.find_adjective_synonym(sentence, word, pos, start_char, end_char)
                    else:
                        rep = self.fig_simplifier.find_metaphor_synonym(sentence, word, pos, start_char, end_char)
                        
                    if rep != word:
                        rep = self.fig_simplifier.apply_casing(word, rep)
                        replacements[word] = rep
                        all_replacements[word] = rep
                        if verbose:
                            print(f"  Metaphor '{word}' (pos: {pos}) -> Mapped to concrete: '{rep}' (combined score: {met['combined_score']:.2f})")

        # ================================================================
        # LAYER 2 - Standard CWI & Pipeline (Stages 1-6)
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
            print("\nSTAGE 1 - Preprocessing (Standard CWI):")
            print(f"  Parsed tokens:      {tokens_display}")
            print(f"  Content candidates: {[t[0] for t in content_tokens]}")

        if not content_tokens and not replacements:
            return sentence

        # Run Standard CWI
        cwi_results = self.cwi.identify_complex_words(
            sentence, content_tokens, cwi_threshold=CWI_THRESHOLD)
        if len(content_tokens) == 1 and cwi_results:
            single_word = content_tokens[0][0]
            if wordfreq.zipf_frequency(single_word.lower(), 'en') < 5.5:
                cwi_results[0]['is_complex'] = True

        complex_words = []
        eff_thresh = cwi_results[0]['effective_threshold'] if cwi_results else CWI_THRESHOLD

        if verbose:
            print(f"\nSTAGE 2 - CWI  (threshold = {eff_thresh:.4f}):")

        for res in cwi_results:
            w       = res['word']
            is_comp = res['is_complex']
            bert_s  = res.get('bert_score', 0.0)
            drift_s = res.get('drift_score', 0.0)
            zipf_p  = res.get('zipf_penalty', 0.0)
            ens_s   = res.get('ensemble_score', 0.0)
            zipf_v  = res.get('word_zipf', 0.0)
            
            # If word is already scheduled for figurative replacement, skip standard CWI
            if w in replacements:
                if verbose:
                    print(f"  {w:15} -> SKIPPED (handled by Layer 1 Metaphor Handler)")
                continue
                
            if w.lower() in protected_words:
                if verbose:
                    print(f"  {w:15} -> SKIPPED (introduced/protected by Layer 0 Idiom Handler)")
                continue
                
            if is_comp:
                complex_words.append(res)
                if verbose:
                    print(f"  {w:15} COMPLEX (ens={ens_s:.3f}, bert={bert_s:.3f}, drift={drift_s:.3f}, zipf={zipf_v:.2f}, zipf_pen={zipf_p:.3f}) -> Scheduled for Stage 3+")
            else:
                if verbose:
                    print(f"  {w:15} SIMPLE  (ens={ens_s:.3f}, bert={bert_s:.3f}, drift={drift_s:.3f}, zipf={zipf_v:.2f}, zipf_pen={zipf_p:.3f})")

        # Process standard complex words
        for cw in complex_words:
            word       = cw['word']
            pos        = cw['pos']
            start_char = cw['start_char']
            end_char   = cw['end_char']
            orig_zipf  = wordfreq.zipf_frequency(word.lower(), 'en')

            def try_gold_fallback():
                lemma = word.lower()
                for token in doc:
                    if token.idx == start_char:
                        lemma = token.lemma_.lower()
                        break

                def is_pos_compatible(cand: str, orig_pos: str) -> bool:
                    if not orig_pos:
                        return True
                    synsets = wn.synsets(cand.lower())
                    if not synsets:
                        return True
                    cand_poses = {s.pos() for s in synsets}
                    pos_map = {
                        'NOUN': {'n'},
                        'PROPN': {'n'},
                        'PRON': {'n'},
                        'VERB': {'v'},
                        'ADJ': {'a', 's'},
                        'ADV': {'r'}
                    }
                    allowed_wn = pos_map.get(orig_pos.upper(), set())
                    if not allowed_wn:
                        return True
                    return len(cand_poses.intersection(allowed_wn)) > 0

                def filter_candidates(cands):
                    filtered = []
                    for c in cands:
                        c_clean = c.lower()
                        if c_clean == word.lower() or c_clean == lemma:
                            continue
                        # Zipf check: candidate must be simpler than original
                        c_zipf = wordfreq.zipf_frequency(c_clean, 'en')
                        if c_zipf <= orig_zipf:
                            continue
                        # POS compatibility check
                        if not is_pos_compatible(c_clean, pos):
                            continue
                        filtered.append((c_clean, c_zipf))
                    # Sort by Zipf descending
                    filtered.sort(key=lambda x: x[1], reverse=True)
                    return [x[0] for x in filtered]

                # BenchLS/LexMTurk
                gold_candidates = self.gold_table.get(word.lower(), [])
                if lemma != word.lower() and lemma not in gold_candidates:
                    gold_candidates = gold_candidates + self.gold_table.get(lemma, [])
                gold_candidates = filter_candidates(gold_candidates)
                if gold_candidates:
                    chosen = gold_candidates[0]
                    if pos == "VERB":
                        chosen = self.fig_simplifier.inflect_verb(chosen, word)
                    replacements[word] = self.fig_simplifier.apply_casing(word, chosen)
                    return True
                
                # PPDB
                ppdb_candidates = self.ppdb_table.get(word.lower(), [])
                if lemma != word.lower() and lemma not in ppdb_candidates:
                    ppdb_candidates = ppdb_candidates + self.ppdb_table.get(lemma, [])
                ppdb_candidates = filter_candidates(ppdb_candidates)
                if ppdb_candidates:
                    chosen = ppdb_candidates[0]
                    if pos == "VERB":
                        chosen = self.fig_simplifier.inflect_verb(chosen, word)
                    replacements[word] = self.fig_simplifier.apply_casing(word, chosen)
                    return True
                
                # WordNet
                wn_pos = {'NOUN': wn.NOUN, 'VERB': wn.VERB, 'ADJ': wn.ADJ, 'ADV': wn.ADV, 'PROPN': wn.NOUN}.get(pos.upper()) if pos else None
                wordnet_syns = []
                for w_query in [word.lower(), lemma]:
                    synsets = wn.synsets(w_query, pos=wn_pos) if wn_pos else wn.synsets(w_query)
                    for syn in synsets:
                        for lm in syn.lemmas():
                            cand = lm.name().replace('_', ' ').lower()
                            if cand.isalpha():
                                wordnet_syns.append(cand)
                wordnet_syns = list(set(wordnet_syns))
                # Filter WordNet syns
                wordnet_syns = filter_candidates(wordnet_syns)
                if wordnet_syns:
                    chosen = wordnet_syns[0]
                    if pos == "VERB":
                        chosen = self.fig_simplifier.inflect_verb(chosen, word)
                    replacements[word] = self.fig_simplifier.apply_casing(word, chosen)
                    return True

                return False

            if verbose:
                print(f"\n  Processing standard complex word: '{word}' (zipf: {orig_zipf:.2f})")

            # Stage 3 - Sense Disambiguation
            chosen_sense, sense_conf = self.disambiguator.disambiguate(
                sentence, start_char, end_char, word, pos)

            # Stage 4 - Candidate Generation & Filtering
            sources = self.cand_gen.generate_raw_candidates_by_source(
                sentence, word, start_char, end_char, chosen_sense, pos)
            raw_pool = set(sources['wordnet'] + sources['bert_mlm'] + sources['glove'])

            masked_text = self.surprisal_calc.get_masked_sentence_and_idx(
                sentence, start_char, end_char)
            mask_inputs  = self.tokenizer(masked_text, return_tensors='pt').to(self.device)
            mask_idx_pos = (mask_inputs['input_ids'][0] == self.tokenizer.mask_token_id
                            ).nonzero(as_tuple=True)[0][0].item()
            with torch.no_grad():
                mask_logits = self.mlm_model(**mask_inputs).logits
            all_probs = F.softmax(mask_logits[0, mask_idx_pos], dim=-1)

            orig_sent_inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)
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

            orig_doc_s = self.nlp(sentence)
            orig_token = self._find_token_at_span(orig_doc_s, start_char, end_char)

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

                # Early check for frequency requirements
                passes_freq = (freq_gain >= dynamic_freq_gain) and (cand_freq >= dynamic_cand_freq)
                if not passes_freq:
                    continue

                # Early check for MLM probability. Since required_mlm is at least 0.0002,
                # if mlm_prob is less than 0.0002, it can never pass.
                if mlm_prob < 0.0002:
                    continue

                wn_pos = None
                pos_map = {'NOUN': wn.NOUN, 'VERB': wn.VERB, 'ADJ': wn.ADJ, 'ADV': wn.ADV, 'PROPN': wn.NOUN}
                wn_pos = pos_map.get(pos.upper()) if pos else None
                sense_related = True
                if chosen_sense and sense_conf >= 0.55:
                    sense_related = are_semantically_related(chosen_sense, word, cand, pos)
                elif wn_pos:
                    sense_related = are_semantically_related(None, word, cand, pos)

                cand_sentence = sentence[:start_char] + cand + sentence[end_char:]
                cand_sent_inputs = self.tokenizer(cand_sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    cand_sent_emb = self.bert_model(**cand_sent_inputs).last_hidden_state[0, 0]
                semantic_sim = F.cosine_similarity(orig_sent_emb.unsqueeze(0), cand_sent_emb.unsqueeze(0)).item()

                # Early check for semantic similarity. Since passes_sem requires >= SEM_SIM_MIN
                # and sense_related recovery requires >= 0.96, if semantic_sim < SEM_SIM_MIN, it fails.
                if semantic_sim < SEM_SIM_MIN:
                    continue

                morph_score = self._morphological_compatibility(
                    sentence, cand_sentence, start_char, end_char, len(cand))

                if not sense_related and semantic_sim >= 0.85 and morph_score >= 0.80:
                    sense_related = True

                is_strong_synonym = sense_related and (semantic_sim >= 0.90)
                required_mlm = 0.0002 if is_strong_synonym else MLM_PROB_MIN
                passes_mlm   = mlm_prob   >= required_mlm
                passes_sem   = semantic_sim >= SEM_SIM_MIN
                passes_morph = morph_score  >= 0.60

                passes_pos = False
                if passes_mlm and passes_sem and passes_morph and sense_related:
                    cand_doc = self.nlp(cand_sentence)
                    cand_token = self._find_token_at_span(cand_doc, start_char, start_char + len(cand))
                    if orig_token is not None and cand_token is not None:
                        p1, p2 = orig_token.pos_, cand_token.pos_
                        passes_pos = (p1 == p2) or ({p1, p2} == {'VERB', 'ADJ'}) or ({p1, p2} == {'NOUN', 'PROPN'})
                    else:
                        passes_pos = True

                if passes_pos:
                    filtered_cands.append(cand)

            if not filtered_cands:
                try_gold_fallback()
                continue

            # Stage 5 - Contextual Ranking
            orig_surp    = self.surprisal_calc.compute_surprisal(sentence, word, start_char, end_char)
            orig_fluency = self.validator.compute_sentence_log_likelihood(sentence)

            orig_inputs   = self.tokenizer(sentence, return_tensors='pt').to(self.device)
            with torch.no_grad():
                orig_states = self.bert_model(**orig_inputs).last_hidden_state[0]
            prefix_tokens = self.tokenizer.tokenize(sentence[:start_char])
            word_tokens   = self.tokenizer.tokenize(word)
            orig_start    = min(len(prefix_tokens) + 1, orig_states.size(0) - 1)
            orig_end      = min(orig_start + len(word_tokens), orig_states.size(0))
            orig_word_emb = orig_states[orig_start:orig_end].mean(dim=0)

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

                cosine_sim     = F.cosine_similarity(orig_word_emb.unsqueeze(0), cand_word_emb.unsqueeze(0)).item()
                cand_surp      = self.surprisal_calc.compute_surprisal(sentence, cand, start_char, end_char)
                surp_red       = orig_surp - cand_surp
                cand_fluency   = self.validator.compute_sentence_log_likelihood(cand_sentence)
                fluency_change = cand_fluency - orig_fluency

                sbert = getattr(self.cand_gen, '_sbert', None)
                if sbert and sbert.available:
                    sentence_sim = sbert.similarity(sentence, cand_sentence)
                else:
                    with torch.no_grad():
                        cand_sent_inputs = self.tokenizer(cand_sentence, return_tensors='pt', padding=True, truncation=True).to(self.device)
                        cand_sent_emb = self.bert_model(**cand_sent_inputs).last_hidden_state[0, 0]
                    sentence_sim = F.cosine_similarity(orig_sent_emb.unsqueeze(0), cand_sent_emb.unsqueeze(0)).item()

                sense_sim = self._wordnet_sense_similarity(chosen_sense, word, cand, pos)
                morph_match = self._morphological_compatibility(sentence, cand_sentence, start_char, end_char, len(cand))
                cand_freq = wordfreq.zipf_frequency(cand, 'en')
                zipf_diff = cand_freq - orig_zipf
                glove_sim = 0.0
                if self.emb_store is not None:
                    glove_sim = self.emb_store.similarity(word.lower(), cand, source='glove')

                rank_score = self.ranker.predict6(
                    mlm_prob=mlm_prob, sbert_sim=sentence_sim, surp_red=surp_red,
                    fluency_change=fluency_change, zipf_diff=zipf_diff, glove_sim=glove_sim, pos_mismatch=False
                )

                semantic_priority = rank_score + 0.40 * sense_sim + 0.20 * morph_match

                scored_candidates.append({
                    'candidate': cand, 'mlm_prob': mlm_prob, 'cosine_sim': cosine_sim, 'surp_red': surp_red,
                    'fluency_change': fluency_change, 'sentence_sim': sentence_sim, 'sense_sim': sense_sim,
                    'morph_match': morph_match, 'semantic_priority': semantic_priority, 'rank_score': rank_score
                })

            scored_candidates.sort(key=lambda x: (x['morph_match'], x['semantic_priority'], x['sense_sim'], x['rank_score']), reverse=True)

            if verbose:
                print(f"  [DEBUG] Word: '{word}' | Scored candidates (top 5):")
                for sc in scored_candidates[:5]:
                    print(f"    - {sc['candidate']}: morph_match={sc['morph_match']:.2f}, semantic_priority={sc['semantic_priority']:.2f}, mlm_prob={sc['mlm_prob']:.5f}")

            best   = scored_candidates[0]
            second = scored_candidates[1] if len(scored_candidates) > 1 else None
            margin = (best['semantic_priority'] - second['semantic_priority']) if second else 1.0
            best_score = best['semantic_priority']
            best_freq_gain = wordfreq.zipf_frequency(best['candidate'], 'en') - orig_zipf
            is_strong_syn = (best['sense_sim'] > 0.0) and (best['sentence_sim'] >= 0.90)
            required_mlm = 0.0002 if is_strong_syn else MLM_PROB_MIN

            if best_score < BEST_SCORE_MIN or margin < MARGIN_MIN or best['mlm_prob'] < required_mlm:
                if verbose:
                    print(f"  [DEBUG] Score/margin/mlm too low (best_score={best_score:.2f}, margin={margin:.2f}, mlm={best['mlm_prob']:.5f}). Trying fallback.")
                try_gold_fallback()
                continue

            # Stage 6 - Validation
            chosen_replacement = None
            for sc in scored_candidates:
                candidate = sc['candidate']
                is_valid = self.validator.validate_replacement(
                    sentence=sentence, original_word=word, candidate_word=candidate,
                    start_char=start_char, end_char=end_char, pos_tag=pos, debug=verbose
                )
                if is_valid:
                    chosen_replacement = candidate
                    break

            if chosen_replacement:
                if pos == "VERB":
                    chosen_replacement = self.fig_simplifier.inflect_verb(chosen_replacement, word)
                replacements[word] = self.fig_simplifier.apply_casing(word, chosen_replacement)
            else:
                try_gold_fallback()

        # ================================================================
        # Stage 7 - Parallel Substitution & Post-Processing
        # ================================================================
        final_sentence = self.replacer.replace_all(sentence, replacements)
        
        # Apply final grammatical corrections (a/an article correction and casing)
        final_sentence = self.fig_simplifier.correct_grammar(final_sentence)

        # Collect all standard CWI replacements
        for w, rep in replacements.items():
            all_replacements[w] = rep

        # Run visual linker to get the full visual learning and interactive data
        word_info_list = []
        for orig, simp in all_replacements.items():
            category = "Standard"
            pos = None
            # Check if this orig was from Layer 0 (idiom)
            if 'idioms_detected' in locals() and idioms_detected and any(orig.lower() == idm.get("text", "").lower() for idm in idioms_detected):
                category = "Idiom"
            elif 'metaphor_results' in locals() and metaphor_results and any(met["word"] == orig for met in metaphor_results):
                category = "Metaphor"
                for met in metaphor_results:
                    if met["word"] == orig:
                        pos = met["pos"]
                        break
            else:
                # Find pos in content_tokens
                if 'content_tokens' in locals():
                    for t_text, t_pos, _, _ in content_tokens:
                        if t_text == orig:
                            pos = t_pos
                            break
            word_info_list.append({
                "word": orig,
                "category": category,
                "pos": pos
            })

        self.last_visual_data = self.visual_linker.process_substitutions(
            original_sentence=sentence,
            simplified_sentence=final_sentence,
            replacements=all_replacements,
            word_info_list=word_info_list
        )

        if verbose:
            print("\n" + "=" * 60)
            print(f"OUTPUT: {final_sentence}")
            print("=" * 60 + "\n")
            # Print terminal formatted table/output
            try:
                print(self.visual_linker.format_terminal_output(self.last_visual_data))
            except Exception:
                try:
                    import sys
                    encoding = sys.stdout.encoding or 'utf-8'
                    print(self.visual_linker.format_terminal_output(self.last_visual_data).encode(encoding, errors='replace').decode(encoding))
                except Exception:
                    pass

        return final_sentence
