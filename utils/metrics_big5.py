# utils/metrics_big5.py
# ------------------------------------------------------
# Extract Big Five personality features (sample mode)
# ------------------------------------------------------

import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

def load_big5_model():
    """Load pretrained BERT model and tokenizer for Big Five personality prediction."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
    model.eval()
    return tokenizer, model

def predict_big5(text, tokenizer, model):
    """Predict Big Five scores (0~1) for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()
    return dict(zip(TRAITS, probs))

def extract_big5_features(dataset_dir, label, save_path, pattern=".txt"):
    """
    Recursively traverse dataset_dir and extract Big5 features from files matching pattern.
    label: 'human' or 'llm'
    pattern: filename suffix (default='.txt', e.g., '_01.txt' for sampling)
    """
    tokenizer, model = load_big5_model()
    records = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(pattern):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    scores = predict_big5(text, tokenizer, model)
                    record = {"filename": file, "path": file_path, "label": label}
                    record.update(scores)
                    records.append(record)
                except Exception as e:
                    print(f"[Error] {file}: {e}")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Big5 features saved to {save_path} ({len(df)} files)")
    return df
