# utils/metrics_nela.py
# ------------------------------------------------------
# Extract emotional + stylistic (NELA-style) features (robust full dataset version)
# ------------------------------------------------------

import os
import re
import pandas as pd
import numpy as np
from textblob import TextBlob
import textstat
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK data available
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


def compute_nela_features(text):
    """Compute simplified NELA-style emotional + stylistic features with safe fallbacks."""
    try:
        # --- 🔹 전처리: 특수문자, 숫자 제거 & 공백 정리 ---
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text.strip())

        # 너무 짧은 텍스트는 무시 (NELA가 정상 계산 불가)
        if len(text.split()) < 10:
            return {
                "num_words": 0, "avg_sentence_length": 0, "word_diversity": 0,
                "polarity": 0, "subjectivity": 0,
                "flesch_reading_ease": 0, "gunning_fog": 0,
                "function_word_ratio": 0, "content_word_ratio": 0
            }

        # --- 🔹 감정 + 언어 feature 계산 ---
        blob = TextBlob(text)
        words = word_tokenize(text)
        stops = set(stopwords.words("english"))

        num_words = len(words)
        num_sentences = len(blob.sentences) or 1
        avg_sentence_len = num_words / (num_sentences + 1e-5)
        word_diversity = len(set(words)) / (num_words + 1e-5)

        # 감정 (TextBlob)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # 읽기 난이도 (Textstat)
        try:
            flesch = textstat.flesch_reading_ease(text)
            fog = textstat.gunning_fog(text)
        except Exception:
            flesch, fog = np.nan, np.nan

        # 기능어/내용어 비율
        stopword_ratio = sum(w.lower() in stops for w in words) / (num_words + 1e-5)
        function_word_ratio = stopword_ratio
        content_word_ratio = 1 - function_word_ratio

        return {
            "num_words": num_words,
            "avg_sentence_length": avg_sentence_len,
            "word_diversity": word_diversity,
            "polarity": polarity,
            "subjectivity": subjectivity,
            "flesch_reading_ease": flesch,
            "gunning_fog": fog,
            "function_word_ratio": function_word_ratio,
            "content_word_ratio": content_word_ratio,
        }

    except Exception as e:
        print(f"[NELA Error: {e}]")
        return {
            "num_words": 0, "avg_sentence_length": 0, "word_diversity": 0,
            "polarity": 0, "subjectivity": 0,
            "flesch_reading_ease": 0, "gunning_fog": 0,
            "function_word_ratio": 0, "content_word_ratio": 0
        }


def extract_nela_features(dataset_dir, label, save_path):
    """
    Recursively traverse dataset_dir and extract NELA-like features from all .txt files.
    label: 'human' or 'llm'
    """
    records = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    # --- 🔹 파일 읽기: UTF-8 → latin-1 fallback ---
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                    except UnicodeDecodeError:
                        with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
                            text = f.read()

                    if len(text.strip()) == 0:
                        continue  # 빈 파일은 스킵

                    feats = compute_nela_features(text)
                    record = {"filename": file, "path": file_path, "label": label}
                    record.update(feats)
                    records.append(record)

                except Exception as e:
                    print(f"[Error reading {file_path}]: {e}")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ NELA features saved to {save_path} ({len(df)} files)")
    return df
