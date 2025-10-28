import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

def load_big5_model():
    # Use pre-trained Big Five personality prediction model
    model_name = "Minej/bert-base-personality"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def predict_big5(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        # Get probabilities using softmax for classification
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    return dict(zip(TRAITS, probs))

def extract_big5_features(dataset_dir, label, save_path):
    tokenizer, model = load_big5_model()
    records = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".txt"):
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
    print(f"Big Five features saved to {save_path} ({len(df)} files)")
    return df
