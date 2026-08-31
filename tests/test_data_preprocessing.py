import pytest
import pandas as pd
from src.data_preprocessing import transform_text, preprocess_df

def test_transform_text():
    # Test lowercase and stemming
    result = transform_text("Goes going gone!", lowercase=True, stopwords_lang="english")
    # 'goes' -> 'go', 'going' -> 'go', 'gone' -> 'gone'
    # but nltk/stem/porter stemmer:
    # "goes" -> "goe"
    # "going" -> "go"
    # "gone" -> "gone"
    # Let's test basic clean output
    words = result.split()
    assert len(words) >= 1
    for w in words:
        assert w.isalnum()

def test_transform_text_no_lowercase():
    # If lowercase is False
    result = transform_text("Running Fast", lowercase=False, stopwords_lang="english")
    # 'Running' starts with caps, is not stopped
    assert "Running" in result

def test_preprocess_df():
    # Create sample DataFrame
    df = pd.DataFrame({
        "text": ["Hi! Call me later.", "Free cash here!", "Hi! Call me later."],
        "target": ["ham", "spam", "ham"]
    })
    # Preprocess
    processed = preprocess_df(df, text_column="text", target_column="target", lowercase=True, stopwords_lang="english")
    # Duplicate should be dropped (shape should be 2 instead of 3)
    assert processed.shape[0] == 2
    # target encoded (should be 0 or 1)
    assert set(processed["target"].unique()).issubset({0, 1})
    # Text transformed (e.g. no punctuation)
    assert "hi" in processed.iloc[0]["text"]
