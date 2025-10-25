import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

TRAITS = ["Extroversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness"]

def load_big5_model():
    """Load the pre-trained Big Five personality prediction model."""
    tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
    model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")
    model.eval()
    return tokenizer, model

def predict_big5(text, tokenizer, model):
    """Predict Big Five personality traits from text using pre-trained model."""
    if len(text.strip()) == 0:
        return {trait: 0.0 for trait in TRAITS}
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.squeeze().detach().numpy()
    
    return {TRAITS[i]: float(predictions[i]) for i in range(len(TRAITS))}

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
