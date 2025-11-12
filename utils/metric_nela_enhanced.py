"""
Enhanced NELA feature extraction with length-independent features.

Extracts:
- Emotional Layer (6 features):
  - polarity, subjectivity (TextBlob)
  - vader_compound, vader_pos, vader_neu, vader_neg (VADER)

- Stylistic Layer (7 features):
  - word_diversity (Type-Token Ratio)
  - flesch_reading_ease (Readability)
  - average_word_length (replaces num_words)
  - sentence_density (replaces avg_sentence_length)
  - punctuation_density
  - smog_index (replaces gunning_fog)
  - verb_ratio (replaces function_word_ratio and content_word_ratio)

All features are length-independent (standardized ratios or sentence-level measures).
"""
import os
import re
import pandas as pd
import nltk
from collections import Counter
from textstat import flesch_reading_ease, smog_index
from textblob import TextBlob

# Download required NLTK data
# Download punkt and punkt_tab (newer NLTK versions prefer punkt_tab)
# Try to download punkt_tab first (newer NLTK versions), then fallback to punkt
_nltk_data_downloaded = False

def _ensure_nltk_data():
    """Ensure all required NLTK data is downloaded."""
    global _nltk_data_downloaded
    if _nltk_data_downloaded:
        return
    
    import warnings
    import sys
    from io import StringIO
    
    # Suppress NLTK download messages temporarily
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    
    try:
        # Try to download punkt_tab first (preferred by newer NLTK)
        for pkg in ['punkt_tab', 'punkt']:
            try:
                nltk.data.find(f'tokenizers/{pkg}')
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
        
        # Download other required packages
        for resource in [
            'taggers/averaged_perceptron_tagger',
            'taggers/averaged_perceptron_tagger_eng',  # Newer NLTK versions
            'vader_lexicon'
        ]:
            try:
                nltk.data.find(resource)
            except LookupError:
                pkg_name = resource.split('/')[-1]
                try:
                    nltk.download(pkg_name, quiet=True)
                except Exception:
                    pass
    finally:
        sys.stderr = old_stderr
    
    _nltk_data_downloaded = True

# Initialize NLTK data
_ensure_nltk_data()

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer (reuse across calls)
_vader_analyzer = None

def get_vader_analyzer():
    """Get or create VADER sentiment analyzer (singleton)."""
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def compute_nela_features_enhanced(text: str) -> dict:
    """
    Compute enhanced NELA features from text.
    
    Returns dict with 13 features:
    Emotional Layer (6):
    - polarity, subjectivity (TextBlob)
    - vader_compound, vader_pos, vader_neu, vader_neg (VADER)
    
    Stylistic Layer (7):
    - word_diversity, flesch_reading_ease, average_word_length,
      sentence_density, punctuation_density, smog_index, verb_ratio
    """
    if not text or len(text.strip()) == 0:
        return {
            # Emotional Layer
            'polarity': 0.0,
            'subjectivity': 0.0,
            'vader_compound': 0.0,
            'vader_pos': 0.0,
            'vader_neu': 0.0,
            'vader_neg': 0.0,
            # Stylistic Layer
            'word_diversity': 0.0,
            'flesch_reading_ease': 0.0,
            'average_word_length': 0.0,
            'sentence_density': 0.0,
            'punctuation_density': 0.0,
            'smog_index': 0.0,
            'verb_ratio': 0.0,
        }
    
    try:
        # Basic tokenization with fallback
        try:
            words = word_tokenize(text.lower())
        except LookupError:
            # Fallback if punkt_tab is not available
            words = text.lower().split()
        words = [w for w in words if w.isalnum()]  # Remove punctuation
        num_words = len(words)
        
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback: split by sentence-ending punctuation
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences) if sentences else 1
        
        # ============================================================
        # EMOTIONAL LAYER (6 features)
        # ============================================================
        
        # 1-2. TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # 3-6. VADER sentiment
        vader_analyzer = get_vader_analyzer()
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']
        vader_pos = vader_scores['pos']
        vader_neu = vader_scores['neu']
        vader_neg = vader_scores['neg']
        
        # ============================================================
        # STYLISTIC LAYER (7 features)
        # ============================================================
        
        # 1. word_diversity (Type-Token Ratio)
        unique_words = len(set(words))
        word_diversity = unique_words / num_words if num_words > 0 else 0.0
        
        # 2. flesch_reading_ease
        flesch_reading_ease_score = flesch_reading_ease(text)
        
        # 3. average_word_length (replaces num_words)
        total_chars = len(text.replace(' ', ''))
        average_word_length = total_chars / num_words if num_words > 0 else 0.0
        
        # 4. sentence_density (replaces avg_sentence_length)
        sentence_density = num_sentences / num_words if num_words > 0 else 0.0
        
        # 5. punctuation_density
        punctuation_count = sum(1 for char in text if char in '.,!?;:()[]{}"\'')
        punctuation_density = punctuation_count / num_words if num_words > 0 else 0.0
        
        # 6. smog_index (replaces gunning_fog)
        smog_index_score = smog_index(text)
        
        # 7. verb_ratio (replaces function_word_ratio and content_word_ratio)
        try:
            pos_tags = pos_tag(words)
            verb_count = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
            verb_ratio = verb_count / num_words if num_words > 0 else 0.0
        except LookupError:
            # If POS tagger is not available, try to download it and retry
            try:
                nltk.download('averaged_perceptron_tagger_eng', quiet=True)
                pos_tags = pos_tag(words)
                verb_count = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
                verb_ratio = verb_count / num_words if num_words > 0 else 0.0
            except Exception:
                # If still fails, set verb_ratio to 0.0
                verb_ratio = 0.0
        
        # ============================================================
        # Return features
        # ============================================================
        
        features = {
            # Emotional Layer (6)
            'polarity': polarity,
            'subjectivity': subjectivity,
            'vader_compound': vader_compound,
            'vader_pos': vader_pos,
            'vader_neu': vader_neu,
            'vader_neg': vader_neg,
            # Stylistic Layer (7)
            'word_diversity': word_diversity,
            'flesch_reading_ease': flesch_reading_ease_score,
            'average_word_length': average_word_length,
            'sentence_density': sentence_density,
            'punctuation_density': punctuation_density,
            'smog_index': smog_index_score,
            'verb_ratio': verb_ratio,
        }
        
        return features
        
    except LookupError as e:
        # LookupError usually means missing NLTK data - try to download and retry
        import nltk
        try:
            # Try to download the missing resource
            if 'averaged_perceptron_tagger' in str(e) or 'tagger' in str(e):
                nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            # Retry the computation
            return compute_nela_features_enhanced(text)
        except Exception:
            # If retry fails, return zeros
            print(f"[NELA Enhanced LookupError] {e}")
            return {
                'polarity': 0.0, 'subjectivity': 0.0,
                'vader_compound': 0.0, 'vader_pos': 0.0, 'vader_neu': 0.0, 'vader_neg': 0.0,
                'word_diversity': 0.0, 'flesch_reading_ease': 0.0, 'average_word_length': 0.0,
                'sentence_density': 0.0, 'punctuation_density': 0.0, 'smog_index': 0.0, 'verb_ratio': 0.0,
            }
    except Exception as e:
        # Print other exceptions for debugging
        print(f"[NELA Enhanced Error] {e}")
        import traceback
        traceback.print_exc()
        return {
            # Emotional Layer
            'polarity': 0.0,
            'subjectivity': 0.0,
            'vader_compound': 0.0,
            'vader_pos': 0.0,
            'vader_neu': 0.0,
            'vader_neg': 0.0,
            # Stylistic Layer
            'word_diversity': 0.0,
            'flesch_reading_ease': 0.0,
            'average_word_length': 0.0,
            'sentence_density': 0.0,
            'punctuation_density': 0.0,
            'smog_index': 0.0,
            'verb_ratio': 0.0,
        }


def extract_nela_features_enhanced_by_model_level(dataset_dir, label, save_path, model, level):
    """
    Extract enhanced NELA features for a specific model and level only.
    
    Args:
        dataset_dir: Root directory containing .txt files
        label: Label for the dataset ("human" or "llm")
        save_path: Path to save the output CSV
        model: Model name (e.g., "LMK", "G12B", "G4B", "DS") - ignored for human
        level: Level name (e.g., "LV1", "LV2", "LV3") - ignored for human
    
    Returns:
        DataFrame with features and metadata
    """
    records = []
    total_files = 0
    is_llm = (label.lower() == "llm")
    
    from utils.parse_dataset_filename import parse_filename
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.lower().endswith(".txt"):
                continue
            
            # Filter by model and level for LLM files
            if is_llm:
                # Check if file matches the target model and level
                if f"_{model}_{level}.txt" not in file:
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
                
                # Extract enhanced NELA features
                feats = compute_nela_features_enhanced(text)
                
                # Parse filename to extract metadata
                meta = parse_filename(file, is_llm=is_llm)
                
                # Build record with metadata
                record = {
                    "filename": file,
                    "path": file_path,
                    "label": label
                }
                
                # Add parsed metadata if available
                if meta:
                    record.update({
                        "domain": meta.get("domain"),
                        "field": meta.get("field"),
                        "author_id": meta.get("author_id"),
                        "year": meta.get("year"),
                        "item_index": meta.get("item_index"),
                        "model": meta.get("model"),
                        "level": meta.get("level")
                    })
                else:
                    # If parsing fails, still add the record but with None metadata
                    record.update({
                        "domain": None,
                        "field": None,
                        "author_id": None,
                        "year": None,
                        "item_index": None,
                        "model": None,
                        "level": None
                    })
                
                # Add enhanced NELA features
                record.update(feats)
                records.append(record)
            except Exception as e:
                print(f"[Error reading {file_path}]: {e}")
    
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    # Print summary
    processed_files = len(records)
    parsed_files = df["domain"].notna().sum()
    print(f"  Enhanced NELA features saved to {save_path}")
    print(f"    Processed: {processed_files} / {total_files} files")
    print(f"    Successfully parsed metadata: {parsed_files} files")
    if processed_files > parsed_files:
        print(f"    Warning: {processed_files - parsed_files} files could not be parsed")
    
    return df


def extract_nela_features_enhanced(dataset_dir, label, save_path):
    """
    Extract enhanced NELA features from all .txt files in dataset_dir.
    
    Args:
        dataset_dir: Root directory containing .txt files (e.g., "dataset/human" or "dataset/llm")
        label: Label for the dataset ("human" or "llm")
        save_path: Path to save the output CSV
    
    Returns:
        DataFrame with features and metadata
    """
    records = []
    total_files = 0
    is_llm = (label.lower() == "llm")
    
    from utils.parse_dataset_filename import parse_filename
    
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
                
                # Extract enhanced NELA features
                feats = compute_nela_features_enhanced(text)
                
                # Parse filename to extract metadata
                meta = parse_filename(file, is_llm=is_llm)
                
                # Build record with metadata
                record = {
                    "filename": file,
                    "path": file_path,
                    "label": label
                }
                
                # Add parsed metadata if available
                if meta:
                    record.update({
                        "domain": meta.get("domain"),
                        "field": meta.get("field"),
                        "author_id": meta.get("author_id"),
                        "year": meta.get("year"),
                        "item_index": meta.get("item_index"),
                        "model": meta.get("model"),
                        "level": meta.get("level")
                    })
                else:
                    # If parsing fails, still add the record but with None metadata
                    record.update({
                        "domain": None,
                        "field": None,
                        "author_id": None,
                        "year": None,
                        "item_index": None,
                        "model": None,
                        "level": None
                    })
                
                # Add enhanced NELA features
                record.update(feats)
                records.append(record)
            except Exception as e:
                print(f"[Error reading {file_path}]: {e}")
    
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    # Print summary
    processed_files = len(records)
    parsed_files = df["domain"].notna().sum()
    print(f"Enhanced NELA features saved to {save_path}")
    print(f"  Processed: {processed_files} / {total_files} files")
    print(f"  Successfully parsed metadata: {parsed_files} files")
    if processed_files > parsed_files:
        print(f"  Warning: {processed_files - parsed_files} files could not be parsed")
    
    return df

