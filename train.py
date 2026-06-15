import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Import local components
from config import CONFIG
from preprocessing import Preprocessor
from contextual_cwi import ComplexWordIdentifier
from dataset import LexicalSimplificationDataset, PrecomputedDataset, precompute_features
from model import LexicalSimplificationModel

def set_seed(seed: int = 42) -> None:
    """
    Sets reproducible random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scheduler: Any, 
    device: torch.device, 
    grad_clip: float
) -> float:
    """
    Performs one training epoch over the precomputed features.
    """
    model.train()
    total_loss = 0.0
    mse_criterion = nn.MSELoss()
    
    for batch in dataloader:
        # Move all tensors to the execution device
        context_embed = batch['context_embed'].to(device)
        mlm_prob = batch['mlm_prob'].to(device)
        semantic_similarity = batch['semantic_similarity'].to(device)
        simplicity_delta = batch['simplicity_delta'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        scores = model(context_embed, mlm_prob, semantic_similarity, simplicity_delta)
        loss = mse_criterion(scores, labels)
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate_epoch(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """
    Validates model performance on the holdout validation set.
    """
    model.eval()
    total_loss = 0.0
    mse_criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in dataloader:
            context_embed = batch['context_embed'].to(device)
            mlm_prob = batch['mlm_prob'].to(device)
            semantic_similarity = batch['semantic_similarity'].to(device)
            simplicity_delta = batch['simplicity_delta'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            scores = model(context_embed, mlm_prob, semantic_similarity, simplicity_delta)
            loss = mse_criterion(scores, labels)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def run_training() -> None:
    """
    Main training routine: sets seeds, loads dataset, precomputes features, trains, validates, and saves.
    """
    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training will run on: {device}")
    
    # 1. Initialize models and tokenizer
    tokenizer = BertTokenizer.from_pretrained(CONFIG['bert_model'])
    
    print("Initializing ranker model...")
    model = LexicalSimplificationModel(CONFIG, tokenizer.vocab_size).to(device)
    
    # Preprocessor and CWI engines
    preprocessor = Preprocessor()
    cwi_engine = ComplexWordIdentifier(CONFIG, tokenizer, model.bert, device)
    
    # 2. Load and parse dataset
    data_dir = "." # Current directory
    try:
        raw_dataset = LexicalSimplificationDataset(CONFIG, tokenizer, data_dir)
        print(f"Loaded {len(raw_dataset)} raw target instances.")
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        sys.exit(1)
        
    # 3. Precompute features to avoid repetitive forward passes
    print("Precomputing features (this may take a few minutes)...")
    precomputed_samples = precompute_features(raw_dataset.samples, CONFIG, tokenizer, model, cwi_engine, device)
    print(f"Finished precomputing features for {len(precomputed_samples)} samples.")
    
    if not precomputed_samples:
        print("Error: No features precomputed. Exiting.")
        sys.exit(1)
        
    full_dataset = PrecomputedDataset(precomputed_samples)
    
    # 4. Train/Val split
    val_size = int(len(full_dataset) * CONFIG['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 5. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr_ranker'], weight_decay=CONFIG['weight_decay'])
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    # 6. Training loop
    best_val_loss = float('inf')
    train_losses: List[float] = []
    val_losses: List[float] = []
    
    print("="*40)
    print("            STARTING TRAINING")
    print("="*40)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, CONFIG['grad_clip'])
        val_loss = validate_epoch(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['best_model_path'])
            print(f" -> Saved best model checkpoint to '{CONFIG['best_model_path']}'")
            
    print(f"\nTraining completed! Best Validation Loss: {best_val_loss:.5f}")
    
    # Plot loss curves and save to disk
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, CONFIG['epochs'] + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, CONFIG['epochs'] + 1), val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Lexical Simplifier Ranking Model Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.close()
    print("Saved loss curves to 'loss_curves.png'")

if __name__ == "__main__":
    run_training()
