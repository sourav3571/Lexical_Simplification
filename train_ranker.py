# train_ranker.py
import os
import torch
from ai_simplifier import AILexicalSimplifier

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'bert_model': 'bert-base-uncased',
        'max_bert_tokens': 128,
    }
    
    print("--- Initializing Lexical Simplifier for Ranker Training ---")
    simplifier = AILexicalSimplifier(config, device=device)
    print("Initialization complete.")
    
    print("\n--- Training GatedFusionRanker on BenchLS & LexMTurk ---")
    simplifier.ranker.train_on_benchls(
        benchls_path='BenchLS.txt',
        lex_mturk_path='lex_mturk.txt',
        tokenizer=simplifier.tokenizer,
        model=simplifier.mlm_model,
        bert_model=simplifier.bert_model,
        device=simplifier.device,
        epochs=5,
        limit=200,
        emb_store=simplifier.emb_store,
        sbert_encoder=getattr(simplifier.cand_gen, '_sbert', None)
    )
    
    # Save the trained ranker weights
    save_path = 'gated_fusion_ranker_6f.pt'
    torch.save(simplifier.ranker.state_dict(), save_path)
    print(f"\n[OK] Saved trained 6-feature ranker weights to {save_path}.")

if __name__ == "__main__":
    main()
