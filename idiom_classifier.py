# idiom_classifier.py

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from typing import List, Dict, Tuple

class IdiomDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item["sentence"]
        phrase = item["phrase"]
        label = float(item["label"])  # float for BCEWithLogitsLoss

        # Tokenize pair: [CLS] sentence [SEP] phrase [SEP]
        encoding = self.tokenizer(
            sentence,
            phrase,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }

class IdiomClassifier(nn.Module):
    """
    RoBERTa-based Idiom Classifier to determine if a phrase is used idiomatic or literal.
    """
    def __init__(self, model_name: str = "roberta-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)

    def predict_idiom_probability(self, sentence: str, phrase: str) -> float:
        """
        Runs inference on a single sentence-phrase pair and returns idiomatic probability.
        """
        self.eval()
        device = next(self.parameters()).device
        encoding = self.tokenizer(
            sentence,
            phrase,
            add_special_tokens=True,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            prob = torch.sigmoid(logits).item()
            
        return prob

def train_idiom_classifier(
    train_path: str = "data/idiom_train.json",
    val_path: str = "data/idiom_val.json",
    model_save_path: str = "idiom_classifier.pt",
    model_name: str = "roberta-base",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_train_samples: int = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Idiom Classifier on {device}...")
    
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load dataset files
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train path {train_path} not found. Please run data_prep.py first.")
        
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Subsetting for fast CPU training if requested or if running on CPU
    if device.type == "cpu" and max_train_samples is None:
        max_train_samples = 150  # Keep it quick on CPU
        print(f"CPU detected: limiting training to {max_train_samples} samples to save time.")
        
    if max_train_samples is not None:
        train_data = train_data[:max_train_samples]
        val_data = val_data[:min(max_train_samples // 5, len(val_data))]

    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    classifier = IdiomClassifier(model_name).to(device)
    tokenizer = classifier.tokenizer
    
    train_dataset = IdiomDataset(train_data, tokenizer)
    val_dataset = IdiomDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(classifier.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = classifier(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct / total if total > 0 else 0
        val_loss, val_acc = evaluate_idiom_classifier(classifier, val_loader, criterion, device)
        
        print(f"Epoch {epoch + 1}: Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), model_save_path)
            print(f"Saved new best model to {model_save_path}")
            
    print("Training finished.")

def evaluate_idiom_classifier(classifier, loader, criterion, device) -> Tuple[float, float]:
    classifier.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = classifier(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    acc = correct / total if total > 0 else 0
    return total_loss / len(loader) if len(loader) > 0 else 0, acc

if __name__ == "__main__":
    # If run directly, run data preparation and then train
    if not os.path.exists("data/idiom_train.json"):
        from data_prep import prepare_all_data
        prepare_all_data()
        
    train_idiom_classifier(epochs=3)
