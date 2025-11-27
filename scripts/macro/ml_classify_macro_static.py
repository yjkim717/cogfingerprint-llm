#!/usr/bin/env python3
"""
Train simple ML classifiers on macro static feature tables (20 dims) to
distinguish human vs LLM (and individual LLM providers).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


DOMAINS = ["news", "blogs", "academic"]
LLM_MODELS = ["DS", "G4B", "G12B", "LMK"]
FEATURE_COLUMNS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
    "polarity",
    "subjectivity",
    "vader_compound",
    "vader_pos",
    "vader_neu",
    "vader_neg",
    "word_diversity",
    "flesch_reading_ease",
    "gunning_fog",
    "average_word_length",
    "num_words",
    "avg_sentence_length",
    "verb_ratio",
    "function_word_ratio",
    "content_word_ratio",
]
HUMAN_MODEL_TAG = "HUMAN"
HUMAN_BASE = Path("macro_dataset/process/human")
LLM_BASE = Path("macro_dataset/process/LLM")
DEFAULT_LEVEL = "LV3"
RESULTS_DIR = Path("results/macro_static_classification")


def parse_year_from_filename(filename: str) -> int | None:
    """Extract 4-digit year token from filename."""
    tokens = filename.replace(".txt", "").split("_")
    for token in tokens:
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_combined_file(path: Path, source: str, model_hint: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = source
    if "model" not in df.columns or df["model"].isna().all():
        df["model"] = model_hint or (HUMAN_MODEL_TAG if source == "human" else "UNKNOWN")
    else:
        df["model"] = df["model"].fillna(model_hint or HUMAN_MODEL_TAG)
    df["model_target"] = df["model"].str.upper()
    df["year"] = df.get("filename", "").apply(parse_year_from_filename)
    df = _coerce_numeric(df, FEATURE_COLUMNS)
    return df


def build_dataset(level: str, verbose: bool = True) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for domain in DOMAINS:
        human_path = HUMAN_BASE / domain / "combined.csv"
        if human_path.exists():
            if verbose:
                print(f"[LOAD] Human {domain}: {human_path}")
            frames.append(load_combined_file(human_path, source="human", model_hint=HUMAN_MODEL_TAG))
        else:
            print(f"[WARN] Missing human combined file: {human_path}")

    for model in LLM_MODELS:
        for domain in DOMAINS:
            llm_path = LLM_BASE / model / level / domain / "combined_outliers_removed.csv"
            if llm_path.exists():
                if verbose:
                    print(f"[LOAD] {model} {domain}: {llm_path}")
                frames.append(load_combined_file(llm_path, source="llm", model_hint=model))
            else:
                print(f"[WARN] Missing LLM combined file: {llm_path}")

    if not frames:
        raise SystemExit("No feature tables loaded. Abort.")

    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].fillna("human").str.lower()
    df["binary_target"] = np.where(df["label"] == "llm", 1, 0)
    df = df.dropna(subset=["year"])
    return df


def split_by_year(df: pd.DataFrame, train_years: Sequence[int], test_years: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = df["year"].isin(train_years)
    test_mask = df["year"].isin(test_years)
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Check year selections.")
    return train_df, test_df


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )


def run_binary_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["binary_target"].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["binary_target"].values

    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    skl_report = classification_report(y_test, preds, target_names=["human", "llm"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(make_pipeline(), X_train, y_train, cv=cv)

    return {
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "classification_report": skl_report,
        "confusion_matrix": cm,
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
    }


def run_multiclass_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df["model_target"])
    test_labels = label_encoder.transform(test_df["model_target"])

    X_train = train_df[FEATURE_COLUMNS].values
    X_test = test_df[FEATURE_COLUMNS].values

    pipeline = make_pipeline()
    pipeline.fit(X_train, train_labels)
    preds = pipeline.predict(X_test)

    report = classification_report(
        test_labels,
        preds,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_labels, preds).tolist()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(make_pipeline(), X_train, train_labels, cv=cv)

    return {
        "classes": label_encoder.classes_.tolist(),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "test_accuracy": float(accuracy_score(test_labels, preds)),
        "classification_report": report,
        "confusion_matrix": cm,
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
    }


def summarize_counts(df: pd.DataFrame) -> dict:
    summary = (
        df.groupby(["label", "model_target", "domain"])
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )
    return {"sample_counts": summary, "total_rows": int(len(df))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Macro static feature ML classification.")
    parser.add_argument("--level", default=DEFAULT_LEVEL, help="LLM level directory to use (default: LV3)")
    parser.add_argument(
        "--train-years",
        nargs="+",
        type=int,
        default=[2020, 2021, 2022, 2023],
        help="Years used for training split.",
    )
    parser.add_argument(
        "--test-years",
        nargs="+",
        type=int,
        default=[2024],
        help="Years used for testing split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "macro_static_classification_summary.json",
        help="Where to save JSON summary.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataset(level=args.level)
    train_df, test_df = split_by_year(df, train_years=args.train_years, test_years=args.test_years)

    binary_results = run_binary_experiment(train_df, test_df)
    multi_results = run_multiclass_experiment(train_df, test_df)

    summary = {
        "settings": {
            "level": args.level,
            "train_years": args.train_years,
            "test_years": args.test_years,
            "feature_columns": FEATURE_COLUMNS,
        },
        "dataset_overview": summarize_counts(df),
        "binary_results": binary_results,
        "multiclass_results": multi_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["binary_results"], indent=2))
    print(json.dumps(summary["multiclass_results"], indent=2))
    print(f"✅ Saved summary to {args.output}")


if __name__ == "__main__":
    main()

