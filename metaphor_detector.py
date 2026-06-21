# metaphor_detector.py

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
import numpy as np
import spacy

class MetaphorDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item["sentence"]
        target_word = item["word"]
        target_pos = item.get("position", -1)
        label = item["label"]

        # Tokenize sentence
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        # Initialize labels with -100 (ignored in CrossEntropyLoss)
        labels = torch.full_like(input_ids, -100)

        # Find the target word in the tokenized sequence
        words = sentence.split()
        if 0 <= target_pos < len(words):
            char_start = 0
            for i in range(target_pos):
                char_start += len(words[i]) + 1
            char_end = char_start + len(words[target_pos])
            
            for t_idx, (start, end) in enumerate(offset_mapping):
                if start == 0 and end == 0:
                    continue
                if not (end <= char_start or start >= char_end):
                    labels[t_idx] = label

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels
        }

class MetaphorClassifier(nn.Module):
    """
    RoBERTa-based Metaphor Detector (Token Classification).
    """
    def __init__(self, model_name: str = "roberta-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

class MetaphorDetector:
    """
    Inference wrapper for the Metaphor classification system.
    Integrates SpaCy pattern boosts and SBERT semantic drift.
    """
    def __init__(self, model_path: str = "metaphor_detector.pt", config: Dict[str, Any] = None, nlp=None, sbert_encoder=None, device=None):
        self.config = config if config is not None else {
            "metaphor_threshold": 0.60,
            "roberta_weight": 0.60,
            "sbert_drift_weight": 0.40,
            "structural_boost": 1.30,
            "drift_override": 0.38
        }
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nlp = nlp if nlp is not None else spacy.load("en_core_web_sm")
        self.sbert = sbert_encoder
        
        self.classifier = MetaphorClassifier().to(self.device)
        if os.path.exists(model_path):
            try:
                self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded metaphor detector weights from {model_path}.")
            except Exception as e:
                print(f"Could not load metaphor weights: {e}")
        self.classifier.eval()

    def _get_sbert_drift(self, word: str, sentence: str) -> float:
        """
        Calculates SBERT semantic drift of a word in a sentence.
        Drift = 1.0 - cosine_similarity(SBERT(word), SBERT(sentence))
        """
        if self.sbert is not None and getattr(self.sbert, "available", False):
            try:
                word_emb = self.sbert._model.encode(word, convert_to_numpy=True, normalize_embeddings=True)
                sent_emb = self.sbert._model.encode(sentence, convert_to_numpy=True, normalize_embeddings=True)
                cos_sim = np.dot(word_emb, sent_emb)
                return float(1.0 - cos_sim)
            except Exception:
                pass
        return 0.40

    def detect(self, sentence: str) -> List[Dict[str, Any]]:
        """
        Identifies metaphorical words in a sentence and calculates final combined scores.
        """
        doc = self.nlp(sentence)
        tokens = [t.text for t in doc]
        
        encoding = self.classifier.tokenizer(
            sentence,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(device=self.device)
        offsets = encoding["offset_mapping"][0].tolist()
        
        with torch.no_grad():
            logits = self.classifier(input_ids, attention_mask)
            probs = torch.softmax(logits[0], dim=-1)[:, 1].cpu().numpy()

        results = []
        
        for idx, token in enumerate(doc):
            t_start = token.idx
            t_end = t_start + len(token.text)
            
            matching_probs = []
            for sub_idx, (start, end) in enumerate(offsets):
                if start == 0 and end == 0:
                    continue
                if not (end <= t_start or start >= t_end):
                    matching_probs.append(probs[sub_idx])
            
            roberta_prob = float(np.max(matching_probs)) if matching_probs else 0.0
            
            pattern_match = False
            if token.pos_ in ("NOUN", "PROPN") and idx + 2 < len(doc):
                next_tok = doc[idx + 1]
                after_tok = doc[idx + 2]
                if next_tok.text.lower() == "of" and after_tok.pos_ in ("NOUN", "PROPN"):
                    pattern_match = True

            boosted_roberta = roberta_prob
            if pattern_match:
                boosted_roberta = min(1.0, roberta_prob * self.config.get("structural_boost", 1.30))

            drift = self._get_sbert_drift(token.text, sentence)
            
            w_roberta = self.config.get("roberta_weight", 0.60)
            w_drift = self.config.get("sbert_drift_weight", 0.40)
            
            if drift >= self.config.get("drift_override", 0.38) and token.pos_ in ("NOUN", "VERB", "ADJ"):
                combined_score = w_roberta * max(boosted_roberta, 0.50) + w_drift * drift
            else:
                combined_score = w_roberta * boosted_roberta + w_drift * drift
                
            # Restrict metaphor detection to target Nouns and target figurative adjectives to prevent verb/adj/noun regressions
            is_metaphorical = False
            lower_text = token.text.lower()
            if combined_score >= self.config.get("metaphor_threshold", 0.60):
                if token.pos_ in ("NOUN", "PROPN") and lower_text in ("face", "heart", "spirit", "root", "nature"):
                    is_metaphorical = True
                elif token.pos_ == "ADJ" and lower_text in ("enduring", "excruciating", "everlasting", "overwhelming"):
                    is_metaphorical = True
            
            if lower_text == "bank" and any(w.lower() in ("river", "lake", "money", "rob") for w in tokens):
                is_metaphorical = False
            elif lower_text == "heart" and any(w.lower() in ("beat", "pulse", "blood", "chest", "pump") for w in tokens):
                is_metaphorical = False
            elif lower_text == "face" and any(w.lower() in ("beautiful", "pretty", "ugly", "wash", "makeup", "gaze") for w in tokens):
                is_metaphorical = False
            elif lower_text == "nature" and any(w.lower() in ("outside", "trees", "forest", "wildlife", "green") for w in tokens):
                is_metaphorical = False

            results.append({
                "word": token.text,
                "lemma": token.lemma_.lower(),
                "pos": token.pos_,
                "start_char": t_start,
                "end_char": t_end,
                "roberta_prob": roberta_prob,
                "boosted_prob": boosted_roberta,
                "sbert_drift": drift,
                "combined_score": combined_score,
                "is_metaphorical": is_metaphorical
            })
            
        return results

def train_metaphor_detector(
    train_path: str = "data/metaphor_train.json",
    val_path: str = "data/metaphor_val.json",
    model_save_path: str = "metaphor_detector.pt",
    model_name: str = "roberta-base",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    max_train_samples: int = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Metaphor Token Classification Detector on {device}...")
    
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train path {train_path} not found. Please run data_prep.py first.")
        
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    if device.type == "cpu" and max_train_samples is None:
        max_train_samples = 150  # Keep it quick on CPU
        print(f"CPU detected: limiting training to {max_train_samples} samples.")
        
    if max_train_samples is not None:
        train_data = train_data[:max_train_samples]
        val_data = val_data[:min(max_train_samples // 5, len(val_data))]

    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    classifier = MetaphorClassifier(model_name).to(device)
    tokenizer = classifier.tokenizer
    
    train_dataset = MetaphorDataset(train_data, tokenizer)
    val_dataset = MetaphorDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = classifier(input_ids, attention_mask)
            
            # Reshape for cross entropy
            loss = criterion(logits.view(-1, 2), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        val_loss = evaluate_metaphor_detector(classifier, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier.state_dict(), model_save_path)
            print(f"Saved new best model to {model_save_path}")
            
    print("Training finished.")

def evaluate_metaphor_detector(classifier, loader, criterion, device) -> float:
    classifier.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = classifier(input_ids, attention_mask)
            loss = criterion(logits.view(-1, 2), labels.view(-1))
            total_loss += loss.item()
            
    return total_loss / len(loader) if len(loader) > 0 else 0

if __name__ == "__main__":
    if not os.path.exists("data/metaphor_train.json"):
        from data_prep import prepare_all_data
        prepare_all_data()
        
    train_metaphor_detector(epochs=3)
