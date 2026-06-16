# generate_augmented_data.py
import re
import json

def main():
    words = [
        "nature", "spirit", "face", "heart", "root", "weight", "depth", "shadow", "core", "fabric",
        "force", "key", "seed", "color", "light", "bridge", "shield", "anchor", "mirror", "path"
    ]

    concepts = [
        "problem", "conflict", "society", "matter", "democracy", "grief", "crisis", "injustice",
        "evil", "resistance", "darkness", "corruption", "war", "truth", "justice", "development",
        "organization", "discussion", "negotiation", "relationship"
    ]

    adjectives = [
        "remarkable", "complex", "clear", "visible", "profound", "evident", "undeniable",
        "significant", "crucial", "essential"
    ]

    literal_adjs = [
        "cold", "warm", "heavy", "light", "large", "small", "wet", "dry", "dirty", "clean"
    ]

    figurative_samples = []
    literal_samples = []

    # Generate 100 figurative samples
    fig_idx = 0
    while len(figurative_samples) < 100:
        word = words[fig_idx % len(words)]
        concept = concepts[(fig_idx * 3) % len(concepts)]
        adj = adjectives[(fig_idx * 7) % len(adjectives)]
        
        # Templates
        templates = [
            f"The {word} of the {concept} is {adj}.",
            f"We must understand the {word} of the {concept}.",
            f"He spoke about the {word} of the {concept}.",
            f"It lies at the very {word} of the {concept}.",
            f"She examined the {word} of the {concept} carefully."
        ]
        
        sentence = templates[fig_idx % len(templates)]
        match = re.search(r'\b' + re.escape(word) + r'\b', sentence, re.IGNORECASE)
        if match:
            figurative_samples.append({
                "sentence": sentence,
                "word": word,
                "start_char": match.start(),
                "end_char": match.end(),
                "label": 1 # complex
            })
        fig_idx += 1

    # Generate 100 literal samples
    lit_idx = 0
    while len(literal_samples) < 100:
        word = words[lit_idx % len(words)]
        lit_adj = literal_adjs[(lit_idx * 5) % len(literal_adjs)]
        
        # Templates
        templates = [
            f"I can see the physical {word} over there.",
            f"The {word} of the object was quite {lit_adj}.",
            f"He touched the rough {word} with his hand.",
            f"Put the {word} on the table, please.",
            f"The clean {word} was placed in the room."
        ]
        
        sentence = templates[lit_idx % len(templates)]
        match = re.search(r'\b' + re.escape(word) + r'\b', sentence, re.IGNORECASE)
        if match:
            literal_samples.append({
                "sentence": sentence,
                "word": word,
                "start_char": match.start(),
                "end_char": match.end(),
                "label": 0 # simple
            })
        lit_idx += 1

    all_samples = figurative_samples + literal_samples
    print(f"Generated {len(all_samples)} samples (Figurative: {len(figurative_samples)}, Literal: {len(literal_samples)})")

    with open("augmented_cwi_data.json", "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2)
    print("Saved to augmented_cwi_data.json")

if __name__ == "__main__":
    main()
