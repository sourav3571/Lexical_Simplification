import sys
import os
import torch
from ai_simplifier import AILexicalSimplifier

# Defined target test cases from the user
TEST_SUITE = [
    # (Category, Input, Expected)
    ("Literal", "The boy went to school.", "The boy went to school."),
    ("Literal", "She ate an apple today.", "She ate an apple today."),
    ("Literal", "The dog ran in the park.", "The dog ran in the park."),
    ("Literal", "He drinks water every day.", "He drinks water every day."),
    ("Literal", "They played football yesterday.", "They played football yesterday."),
    ("Lexical", "The boy was exhausted.", "The boy was tired."),
    ("Lexical", "She purchased a dress.", "She bought a dress."),
    ("Lexical", "He obtained permission.", "He got permission."),
    ("Lexical", "The girl was delighted.", "The girl was happy."),
    ("Lexical", "The man consumed food.", "The man ate food."),
    ("Lexical", "She has innate talent.", "She has natural talent."),
    ("Lexical", "He commenced working.", "He started working."),
    ("Lexical", "She was very fatigued.", "She was very tired."),
    ("Lexical", "He endeavored to win.", "He tried to win."),
    ("Lexical", "She utilized the tool.", "She used the tool."),
    ("Lexical", "The physician treated the patient.", "The doctor treated the patient."),
    ("Lexical", "She demonstrated exceptional courage.", "She showed great courage."),
    ("Lexical", "He acquired sufficient knowledge.", "He gained enough knowledge."),
    ("Lexical", "The medication was administered.", "The medicine was given."),
    ("Lexical", "She exhibited remarkable patience.", "She showed great patience."),
    ("Metaphor", "The nature of the person is good.", "The character of the person is good."),
    ("Metaphor", "The face of poverty is visible.", "The reality of poverty is visible."),
    ("Metaphor", "The heart of the problem is trust.", "The core of the problem is trust."),
    ("Metaphor", "The root of success is hard work.", "The basis of success is hard work."),
    ("Metaphor", "The spirit of the law must be followed.", "The meaning of the law must be followed."),
    ("Metaphor", "The shadow of doubt remained.", "The feeling of doubt remained."),
    ("Metaphor", "The weight of responsibility grew.", "The burden of responsibility grew."),
    ("Metaphor", "The fabric of society is changing.", "The structure of society is changing."),
    ("Metaphor", "The depth of his knowledge showed.", "The level of his knowledge showed."),
    ("Metaphor/Adj", "The strength is enduring.", "The strength is lasting."),
    ("Idiom", "He kicked the bucket last year.", "He died last year."),
    ("Idiom", "She is under the weather today.", "She is feeling sick today."),
    ("Idiom", "He spilled the beans about the plan.", "He revealed the secret about the plan."),
    ("Idiom", "They hit the nail on the head.", "They were exactly right."),
    ("Idiom", "She bit the bullet and went ahead.", "She endured it and went ahead."),
    ("Idiom", "He let the cat out of the bag.", "He revealed the secret."),
    ("Idiom", "They burned the midnight oil.", "They worked very late."),
    ("Idiom", "She beat around the bush.", "She avoided the main point."),
    ("Idiom", "He broke the ice at the meeting.", "He made people comfortable at the meeting."),
    ("Idiom", "It cost an arm and a leg.", "It was very expensive."),
    ("Contextual", "The nature outside is beautiful.", "The nature outside is beautiful."),
    ("Contextual", "The nature of evil is complex.", "The character of evil is complex."),
    ("Contextual", "The bank near the river flooded.", "The bank near the river flooded."),
    ("Contextual", "She runs every morning.", "She runs every morning."),
    ("Contextual", "He runs a large company.", "He manages a large company."),
    ("Contextual", "His heart beats very fast.", "His heart beats very fast."),
    ("Contextual", "The heart of the city is busy.", "The center of the city is busy."),
    ("Contextual", "The face was beautiful.", "The face was beautiful."),
    ("Contextual", "The face of poverty is real.", "The reality of poverty is real."),
    ("Contextual", "The strength is enduring.", "The strength is lasting."),
    ("Lexical/Adj", "The pain was excruciating.", "The pain was severe."),
    ("Lexical/Adj", "The bond is everlasting.", "The bond is permanent."),
    ("Lexical/Adj", "The feeling was overwhelming.", "The feeling was intense."),
    ("Lexical/Adj", "The task was arduous.", "The task was hard."),
    ("Lexical/Adj", "The silence was deafening.", "The silence was very loud."),
    ("Lexical/Adj", "The situation was dire.", "The situation was very bad."),
    ("Lexical/Adj", "The loss was devastating.", "The loss was very bad."),
    ("Lexical/Adj", "The view was breathtaking.", "The view was very beautiful."),
    ("Lexical/Adj", "The cold was bitter.", "The cold was very harsh."),
    ("Lexical/Adj", "The bond between them is everlasting.", "The bond between them is permanent."),
    ("Academic", "The hypothesis was empirically validated.", "The idea was proven by evidence."),
    ("Academic", "The results were statistically significant.", "The results were very important."),
    ("Academic", "The methodology was comprehensive.", "The method was complete."),
    ("Academic", "The phenomenon remains unexplained.", "The event remains unexplained."),
    ("Academic", "The data was meticulously analyzed.", "The data was carefully studied."),
    ("Academic", "The findings contradict previous assumptions.", "The findings disagree with previous beliefs."),
    ("Academic", "The framework facilitates collaboration.", "The system helps teamwork."),
    ("Academic", "The implications were thoroughly analyzed.", "The effects were carefully studied."),
    ("Academic", "The correlation between variables was significant.", "The link between variables was important."),
    ("Academic", "The paradigm shift altered scientific thinking.", "The change altered scientific thinking."),
    ("Medical", "The physician prescribed medication.", "The doctor prescribed medicine."),
    ("Medical", "The patient was administered drugs.", "The patient was given drugs."),
    ("Medical", "The surgery was deemed necessary.", "The operation was seen as needed."),
    ("Medical", "The diagnosis was inconclusive.", "The diagnosis was unclear."),
    ("Medical", "The symptoms were alleviated by treatment.", "The symptoms were reduced by treatment."),
    ("Medical", "The cardiovascular procedure was complex.", "The heart operation was complex."),
    ("Medical", "The neurological assessment revealed abnormalities.", "The brain test revealed problems."),
    ("Medical", "The pharmaceutical company developed a new drug.", "The medicine company developed a new drug."),
    ("Medical", "The immunological response was stronger.", "The immune response was stronger."),
    ("Medical", "The surgical procedure was successful.", "The operation was successful."),
    ("Complex/Mixed", "The administration implemented comprehensive reforms.", "The government made complete changes."),
    ("Complex/Mixed", "The physician recommended a nutritious diet.", "The doctor recommended a healthy diet."),
    ("Complex/Mixed", "She demonstrated exceptional resilience.", "She showed great strength."),
    ("Complex/Mixed", "The situation was increasingly precarious.", "The situation was increasingly dangerous."),
    ("Complex/Mixed", "His benevolent disposition endeared him to everyone.", "His kind nature made everyone like him."),
    ("Complex/Mixed", "She kicked the bucket after a prolonged illness.", "She died after a long illness."),
    ("Complex/Mixed", "He burned the midnight oil to improve his work.", "He worked very late to improve his work."),
    ("Complex/Mixed", "The face of the crisis was becoming precarious.", "The reality of the crisis was becoming dangerous."),
    ("Complex/Mixed", "The strength of their bond was truly everlasting.", "The strength of their bond was truly permanent."),
    ("Complex/Mixed", "The nature of her innate abilities was remarkable.", "The character of her natural abilities was impressive."),
    ("Complex/Mixed", "He utilized sophisticated methodology.", "He used a complex method."),
    ("Complex/Mixed", "The legislation was ratified unanimously.", "The law was approved by everyone."),
    ("Complex/Mixed", "She articulated her argument clearly.", "She explained her point clearly."),
    ("Complex/Mixed", "The ramifications were far reaching.", "The effects were wide ranging."),
    ("Complex/Mixed", "The corporation terminated employment.", "The company ended the jobs."),
    ("Complex/Mixed", "He had an innate ability to lead.", "He had a natural ability to lead."),
    ("Complex/Mixed", "The defendant was acquitted of charges.", "The defendant was cleared of charges."),
    ("Complex/Mixed", "The initiative was well received.", "The plan was well received."),
    ("Complex/Mixed", "She portrayed the character authentically.", "She showed the character honestly."),
    ("Complex/Mixed", "The amelioration of poverty requires reform.", "The improvement of poverty requires change.")
]

def run_verification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'bert_model': 'bert-base-uncased',
        'max_bert_tokens': 128,
        'FIG_CONFIG': {
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
        }
    }
    
    print("--- Initializing Lexical Simplifier ---")
    simplifier = AILexicalSimplifier(config, device=device)
    print("Initialization complete.\n")
    
    passed_count = 0
    total_count = len(TEST_SUITE)
    failures = []
    
    print(f"{'ID':<4} | {'Category':<15} | {'Status':<10} | {'Input Sentence'}")
    print("-" * 80)
    
    for idx, (category, input_sent, expected) in enumerate(TEST_SUITE, 1):
        output = simplifier.simplify(input_sent, verbose=False)
        clean_output = " ".join(output.strip().split())
        clean_expected = " ".join(expected.strip().split())
        
        status = "PASS"
        if clean_output.lower() != clean_expected.lower():
            status = "FAIL"
            failures.append((idx, category, input_sent, clean_expected, clean_output))
        else:
            passed_count += 1
            
        print(f"{idx:03d}  | {category:<15} | {status:<10} | {input_sent}")
        
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS:")
    print("=" * 60)
    print(f"Total Test Cases: {total_count}")
    print(f"Passed:           {passed_count} ({passed_count/total_count*100:.2f}%)")
    print(f"Failed:           {len(failures)} ({len(failures)/total_count*100:.2f}%)")
    print("=" * 60)
    
    if failures:
        print("\nDETAILED FAILURES:")
        print("-" * 80)
        for f_idx, f_cat, f_in, f_exp, f_out in failures:
            print(f"Test Case {f_idx:03d} ({f_cat})")
            print(f"  Input:    \"{f_in}\"")
            print(f"  Expected: \"{f_exp}\"")
            print(f"  Got:      \"{f_out}\"")
            print("-" * 80)

if __name__ == "__main__":
    run_verification()
