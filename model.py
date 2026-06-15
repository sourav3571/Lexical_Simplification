import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForMaskedLM
from typing import Dict, Any

class LexicalSimplificationModel(nn.Module):
    """
    LexicalSimplificationModel ranks candidate simplifications.
    Combines BERT contextual CLS embedding, MLM probabilities, semantic similarity,
    and simplicity delta through projection sub-networks to output a score in [0, 1].
    """
    def __init__(self, config: Dict[str, Any], vocab_size: int) -> None:
        """
        Sets up sub-networks, loads pretrained BERT weights, and initializes the classification layers.
        """
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['bert_model'])
        
        self.fine_tune_bert = config.get('fine_tune_bert', False)
        if not self.fine_tune_bert:
            print("Freezing BERT encoder weights for fast CPU training...")
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.mlm_head = nn.Linear(768, vocab_size)
        print("Initializing MLM head with pretrained BERT weights...")
        try:
            pretrained_mlm = BertForMaskedLM.from_pretrained(config['bert_model'])
            with torch.no_grad():
                # Load weights from pretrained masked LM classification decoder
                self.mlm_head.weight.copy_(pretrained_mlm.cls.predictions.decoder.weight)
                self.mlm_head.bias.copy_(pretrained_mlm.cls.predictions.bias)
            print("MLM head weights successfully loaded.")
            del pretrained_mlm
        except Exception as exc:
            print(f"Could not load pretrained MLM weights: {exc}. Initializing randomly.")
            
        self.cosine = nn.CosineSimilarity(dim=1)
        
        # 1. CLS Context projection (768 -> 16)
        self.context_proj = nn.Linear(768, 16)
        
        # 2. MLM probability projection (1 -> 4)
        self.mlm_proj = nn.Linear(1, 4)
        
        # Combined features ranker: context_proj (16) + mlm_proj (4) + semantic cosine (1) + simplicity delta (1) = 22
        self.ranker_net = nn.Sequential(
            nn.Linear(16 + 4 + 1 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(
        self, 
        context_embed: torch.Tensor, 
        mlm_prob: torch.Tensor, 
        semantic_similarity: torch.Tensor, 
        simplicity_delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Runs candidate scores predictions.
        
        Args:
            context_embed: [batch_size, 768] CLS embeddings
            mlm_prob: [batch_size, 1] candidate MLM probability
            semantic_similarity: [batch_size, 1] cosine similarity
            simplicity_delta: [batch_size, 1] target complexity - candidate complexity
        """
        # Project CLS context
        proj_context = self.context_proj(context_embed) # Shape: [batch_size, 16]
        
        # Project MLM prob
        proj_mlm = self.mlm_proj(mlm_prob) # Shape: [batch_size, 4]
        
        # Concatenate features: 16 + 4 + 1 + 1 = 22-dim representation
        features = torch.cat([proj_context, proj_mlm, semantic_similarity, simplicity_delta], dim=-1)
        
        # Score candidates
        score = self.ranker_net(features) # Shape: [batch_size, 1]
        return score
