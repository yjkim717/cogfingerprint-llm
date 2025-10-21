import os
import re
import pandas as pd
import numpy as np
from textblob import TextBlob
import textstat
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass


def compute_nela_features(text):
    try:
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text.strip())

        if len(text.split()) < 10:
            return {k: 0 for k in [
                "num_words", "avg_sentence_length", "word_diversity",
                "polarity", "subjectivity",
                "flesch_reading_ease", "gunning_fog",
                "function_word_ratio", "content_word_ratio"
            ]}

        blob = TextBlob(text)
        try:
            words = word_tokenize(text)
        except LookupError:
            nltk.download("punkt")
            words = text.split()

        stops = set(stopwords.words("english"))
        num_words = len(words)

        try:
            num_sentences = len(blob.sentences)
        except Exception:
            num_sentences = max(1, text.count("."))

        avg_sentence_len = num_words / (num_sentences + 1e-5)
        word_diversity = len(set(words)) / (num_words + 1e-5)

        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        try:
            flesch = textstat.flesch_reading_ease(text)
            fog = textstat.gunning_fog(text)
        except Exception:
            flesch, fog = np.nan, np.nan

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
        print(f"[NELA Error] {e}")
        return {k: 0 for k in [
            "num_words", "avg_sentence_length", "word_diversity",
            "polarity", "subjectivity",
            "flesch_reading_ease", "gunning_fog",
            "function_word_ratio", "content_word_ratio"
        ]}


def extract_nela_features(dataset_dir, label, save_path):
    records = []
    total_files = 0

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.lower().endswith(".txt"):
                continue
            total_files += 1
            file_path = os.path.join(root, file)
            try:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
                        text = f.read()

                if len(text.strip()) == 0:
                    continue

                feats = compute_nela_features(text)
                record = {"filename": file, "path": file_path, "label": label}
                record.update(feats)
                records.append(record)

            except Exception as e:
                print(f"[Error reading {file_path}]: {e}")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"NELA features saved to {save_path} ({len(df)} / {total_files} files processed)")
    return df
