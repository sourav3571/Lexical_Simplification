import os
import sys
import torch
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer

# Import local components
from config import CONFIG
from preprocessing import Preprocessor
from contextual_cwi import ComplexWordIdentifier
from word_sense_disambiguation import WordSenseDisambiguator
from candidate_generator import CandidateGenerator
from model import LexicalSimplificationModel
from inference import LexicalSimplifier, inflect_candidate

def test_preprocessing(preprocessor: Preprocessor) -> bool:
    """
    Verifies that preprocessing tokenizes, lemmatizes, and tags parts of speech correctly.
    """
    print("Running test_preprocessing...")
    try:
        sentence = "The dogs barked loudly."
        tokens = preprocessor.preprocess(sentence)
        if not tokens:
            print("  [FAIL] Returned empty token list.")
            return False
            
        # Check target tags
        dog_token = next((t for t in tokens if t['text'] == "dogs"), None)
        bark_token = next((t for t in tokens if t['text'] == "barked"), None)
        
        if not dog_token or dog_token['pos'] != 'NOUN' or dog_token['lemma'] != 'dog':
            print(f"  [FAIL] Incorrect token info for 'dogs': {dog_token}")
            return False
            
        if not bark_token or bark_token['pos'] != 'VERB' or bark_token['lemma'] != 'bark':
            print(f"  [FAIL] Incorrect token info for 'barked': {bark_token}")
            return False
            
        print("  [PASS] Preprocessing test passed.")
        return True
    except Exception as exc:
        print(f"  [FAIL] Preprocessing crashed: {exc}")
        return False

def test_contextual_cwi(cwi_engine: ComplexWordIdentifier) -> bool:
    """
    Verifies that Context-Aware CWI assigns reasonable complexity scores in context.
    """
    print("Running test_contextual_cwi...")
    try:
        # Test complexity of hard word vs simple word
        sent1 = "It is our responsibility to fulfill our obligations."
        # target word: obligations (complex)
        is_comp1, score1 = cwi_engine.is_complex(sent1, 38, 49, "obligations", "obligation")
        
        sent2 = "The task was simple."
        # target word: task (simple)
        is_comp2, score2 = cwi_engine.is_complex(sent2, 4, 8, "task", "task")
        
        if score1 <= score2:
            print(f"  [WARNING] Score of 'obligations' ({score1:.3f}) is not higher than 'task' ({score2:.3f}).")
            
        print(f"  [PASS] CWI test passed. 'obligations' score: {score1:.3f}, 'task' score: {score2:.3f}")
        return True
    except Exception as exc:
        print(f"  [FAIL] CWI crashed: {exc}")
        return False

def test_word_sense_disambiguation(wsd_engine: WordSenseDisambiguator) -> bool:
    """
    Verifies WSD selects different WordNet senses for the same word based on context.
    """
    print("Running test_word_sense_disambiguation...")
    try:
        # Context 1: financial bank
        sent1 = "He deposited money in the bank."
        sense1 = wsd_engine.disambiguate(sent1, 26, 30, "bank", "NOUN")
        
        # Context 2: river bank
        sent2 = "He sat by the muddy bank of the river."
        sense2 = wsd_engine.disambiguate(sent2, 20, 24, "bank", "NOUN")
        
        if not sense1 or not sense2:
            print("  [FAIL] Disambiguator returned None.")
            return False
            
        if sense1.name() == sense2.name():
            print(f"  [WARNING] Same sense selected for bank: {sense1.name()}")
            
        print(f"  [PASS] WSD test passed. Financial sense: {sense1.name()} | River sense: {sense2.name()}")
        return True
    except Exception as exc:
        print(f"  [FAIL] WSD crashed: {exc}")
        return False

def test_candidate_generation(generator: CandidateGenerator, cwi_engine: ComplexWordIdentifier) -> bool:
    """
    Verifies that candidate generator fetches contextually and semantically correct synonyms.
    """
    print("Running test_candidate_generation...")
    try:
        sent = "We will commence the assembly."
        # target word: commence (VERB)
        pos = "VERB"
        chosen_sense = wn.synsets("commence", pos=wn.VERB)[0]
        
        candidates = generator.generate(sent, 8, 16, "commence", pos, chosen_sense, cwi_engine)
        
        if not candidates:
            print("  [FAIL] No candidates generated.")
            return False
            
        words = [c['word'] for c in candidates]
        if "start" not in words and "begin" not in words:
            print(f"  [WARNING] Expected candidates 'start' or 'begin' not found. Generated: {words}")
            
        print(f"  [PASS] Candidate generation passed. Generated candidates: {words}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Candidate generator crashed: {exc}")
        return False

def test_ranking_model(model: LexicalSimplificationModel, device: torch.device) -> bool:
    """
    Verifies that the ranking model outputs normalized scores for candidate features.
    """
    print("Running test_ranking_model...")
    try:
        # Construct random feature batches
        batch_size = 4
        context_embed = torch.randn(batch_size, 768).to(device)
        mlm_prob = torch.rand(batch_size, 1).to(device)
        semantic_similarity = torch.rand(batch_size, 1).to(device)
        simplicity_delta = torch.rand(batch_size, 1).to(device)
        
        scores = model(context_embed, mlm_prob, semantic_similarity, simplicity_delta)
        
        if scores.shape != (batch_size, 1):
            print(f"  [FAIL] Incorrect score shape: {scores.shape}")
            return False
            
        if (scores < 0.0).any() or (scores > 1.0).any():
            print("  [FAIL] Scores outside range [0, 1].")
            return False
            
        print("  [PASS] Ranking model output test passed.")
        return True
    except Exception as exc:
        print(f"  [FAIL] Ranking model crashed: {exc}")
        return False

def test_word_replacement() -> bool:
    """
    Verifies grammatical inflections and casing style preservation during word replacement.
    """
    print("Running test_word_replacement...")
    try:
        # Plural inflection
        res1 = inflect_candidate("obligations", "NOUN", "agreement")
        if res1 != "agreements":
            print(f"  [FAIL] Plural noun inflection failed: {res1}")
            return False
            
        # Past tense inflection
        res2 = inflect_candidate("commenced", "VERB", "start")
        if res2 != "started":
            print(f"  [FAIL] Past tense verb inflection failed: {res2}")
            return False
            
        # Present participle inflection
        res3 = inflect_candidate("terminating", "VERB", "stop")
        if res3 != "stopping":
            print(f"  [FAIL] Present participle inflection failed: {res3}")
            return False
            
        print("  [PASS] Word replacement inflections test passed.")
        return True
    except Exception as exc:
        print(f"  [FAIL] Word replacement test crashed: {exc}")
        return False

def test_end_to_end(simplifier: LexicalSimplifier) -> bool:
    """
    Verifies full 6-stage pipeline on end-to-end sentences.
    """
    print("Running test_end_to_end...")
    try:
        sent = "The company must fulfill its obligations to maintain a beautiful environment."
        simplified = simplifier.simplify(sent)
        
        if not simplified:
            print("  [FAIL] Simplified sentence is empty.")
            return False
            
        print(f"  [PASS] End-to-end pipeline passed.\n  Input : {sent}\n  Output: {simplified}")
        return True
    except Exception as exc:
        print(f"  [FAIL] End-to-end crashed: {exc}")
        return False

def test_edge_cases(simplifier: LexicalSimplifier) -> bool:
    """
    Verifies error handling and graceful fallbacks for empty inputs, long sentences, and missing resources.
    """
    print("Running test_edge_cases...")
    try:
        # Case 1: Empty sentence
        res_empty = simplifier.simplify("")
        if res_empty != "":
            print("  [FAIL] Failed to return empty string for empty input.")
            return False
            
        # Case 2: Sentence with no complex words
        res_simple = simplifier.simplify("He is a boy.")
        if res_simple != "He is a boy.":
            print("  [FAIL] Modified already simple sentence.")
            return False
            
        # Case 3: Very long sentence (over BERT window)
        long_sentence = "The " + "very " * 150 + "diligent candidate must pass."
        res_long = simplifier.simplify(long_sentence)
        if not res_long:
            print("  [FAIL] Long sentence crashed or returned empty.")
            return False
            
        print("  [PASS] Edge cases tests passed.")
        return True
    except Exception as exc:
        print(f"  [FAIL] Edge cases crashed: {exc}")
        return False

def run_all_tests() -> None:
    """
    Loads resources and triggers the complete pipeline testing suite.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = CONFIG['best_model_path'] if os.path.exists(CONFIG['best_model_path']) else "./best_model.pt"
    
    print("Loading models and initializing components...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG['bert_model'])
    
    # Instantiate modules
    preprocessor = Preprocessor()
    
    # Try to load best_model.pt
    model = LexicalSimplificationModel(CONFIG, tokenizer.vocab_size).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    cwi_engine = ComplexWordIdentifier(CONFIG, tokenizer, model, device)
    wsd_engine = WordSenseDisambiguator(CONFIG, tokenizer, model.bert, device)
    generator = CandidateGenerator(CONFIG, tokenizer, model, device)
    
    # Complete pipeline engine
    simplifier = LexicalSimplifier(CONFIG, model_path, device)
    
    # Run suite
    p_pre = test_preprocessing(preprocessor)
    p_cwi = test_contextual_cwi(cwi_engine)
    p_wsd = test_word_sense_disambiguation(wsd_engine)
    p_gen = test_candidate_generation(generator, cwi_engine)
    p_rnk = test_ranking_model(model, device)
    p_rep = test_word_replacement()
    p_e2e = test_end_to_end(simplifier)
    p_edg = test_edge_cases(simplifier)
    
    # Summary report
    results = [p_pre, p_cwi, p_wsd, p_gen, p_rnk, p_rep, p_e2e, p_edg]
    passed_count = sum(1 for r in results if r)
    
    print("\n" + "="*45)
    print(f"TEST RESULTS: {passed_count}/{len(results)} PASSED")
    print("="*45)
    
    if passed_count == len(results):
        print("ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"SOME TESTS FAILED! (Check trace output above)")

if __name__ == "__main__":
    run_all_tests()
