import os
import torch
from ai_simplifier import AILexicalSimplifier

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'bert_model': 'bert-base-uncased',
        'max_bert_tokens': 128
    }
    
    print("Initializing AILexicalSimplifier (this may take a moment)...")
    simplifier = AILexicalSimplifier(config, device)
    print("Simplifier ready!\n")
    
    while True:
        try:
            sentence = input("Enter a sentence to simplify (or enter '1' to exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if sentence == '1':
            print("Exiting...")
            break
            
        if not sentence:
            continue
            
        try:
            # Running with verbose=False for clean interactive CLI output
            simplified = simplifier.simplify(sentence, verbose=False)
            print(f"\nResult: {simplified}\n")
        except Exception as e:
            print(f"Error simplifying sentence: {e}\n")

if __name__ == "__main__":
    main()
