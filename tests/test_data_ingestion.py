import pytest
import pandas as pd
import yaml
from unittest.mock import patch, mock_open
from src.data_ingestion import load_params, load_data, preprocess_data

def test_load_params():
    mock_yaml = """
    data_ingestion:
      test_size: 0.2
    """
    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        params = load_params("fake_params.yaml")
        assert params["data_ingestion"]["test_size"] == 0.2

def test_load_params_not_found():
    with pytest.raises(FileNotFoundError):
        load_params("non_existent_params.yaml")

def test_load_data():
    mock_df = pd.DataFrame({"col": [1, 2]})
    with patch("pandas.read_csv", return_value=mock_df):
        df = load_data("fake_url")
        assert df.shape == (2, 1)

def test_preprocess_data():
    # Input has Unnamed: 2, 3, 4 and columns v1, v2
    df = pd.DataFrame({
        "v1": ["ham", "spam"],
        "v2": ["hello", "free tickets"],
        "Unnamed: 2": [None, None],
        "Unnamed: 3": [None, None],
        "Unnamed: 4": [None, None]
    })
    processed = preprocess_data(df)
    assert "Unnamed: 2" not in processed.columns
    assert "target" in processed.columns
    assert "text" in processed.columns
    assert processed.loc[0, "target"] == "ham"
    assert processed.loc[1, "text"] == "free tickets"
