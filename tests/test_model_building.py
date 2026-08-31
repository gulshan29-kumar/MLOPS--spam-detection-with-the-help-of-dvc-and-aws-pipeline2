import pytest
import numpy as np
import pandas as pd
import pickle
import os
from unittest.mock import patch
from src.model_building import load_data, train_model, save_model

def test_load_data():
    df = pd.DataFrame({"feat1": [1.0, 2.0], "label": [1, 2]})
    with patch("pandas.read_csv", return_value=df):
        loaded = load_data("fake_features.csv")
        assert loaded.shape == (2, 2)

def test_train_model():
    X = np.random.rand(10, 5)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    params = {"n_estimators": 5, "random_state": 42}
    
    model = train_model(X, y, params)
    assert model.n_estimators == 5
    assert model.random_state == 42
    
    # Check predictions format
    preds = model.predict(X)
    assert len(preds) == 10

def test_train_model_dimension_mismatch():
    X = np.random.rand(10, 5)
    y = np.array([0, 1, 0])
    params = {"n_estimators": 5, "random_state": 42}
    with pytest.raises(ValueError):
        train_model(X, y, params)

def test_save_model(tmp_path):
    model = "mock_model_object"
    file_path = tmp_path / "model.pkl"
    save_model(model, str(file_path))
    assert os.path.exists(file_path)
    with open(file_path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == "mock_model_object"
