import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch
from src.feature_engineering import load_data, apply_tfidf, save_data

def test_load_data():
    df = pd.DataFrame({"text": ["hello", None], "target": [1, 0]})
    # Mock pd.read_csv to return this df
    with patch("pandas.read_csv", return_value=df):
        loaded = load_data("test_path.csv")
        # Null values should be filled with empty strings
        assert loaded.loc[1, "text"] == ""

def test_apply_tfidf():
    train_data = pd.DataFrame({"text": ["spam text", "ham text"], "target": [1, 0]})
    test_data = pd.DataFrame({"text": ["clean text email"], "target": [0]})
    
    train_df, test_df = apply_tfidf(train_data, test_data, max_features=10)
    
    # Train should have 2 samples
    assert train_df.shape[0] == 2
    # Test should have 1 sample
    assert test_df.shape[0] == 1
    # label column should exist
    assert "label" in train_df.columns
    assert "label" in test_df.columns
    assert train_df.loc[0, "label"] == 1
    assert test_df.loc[0, "label"] == 0

def test_save_data(tmp_path):
    df = pd.DataFrame({"col": [1, 2]})
    file_path = tmp_path / "subdir" / "output.csv"
    save_data(df, str(file_path))
    assert os.path.exists(file_path)
    loaded = pd.read_csv(file_path)
    assert loaded.shape == (2, 1)
