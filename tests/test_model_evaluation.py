import pytest
import numpy as np
import pandas as pd
import os
import json
from unittest.mock import patch, MagicMock
from src.model_evaluation import load_model, load_data, evaluate_model, save_metrics

def test_load_model():
    with patch("builtins.open", MagicMock()):
        with patch("pickle.load", return_value="mock_clf"):
            clf = load_model("mock_model.pkl")
            assert clf == "mock_clf"

def test_load_data():
    df = pd.DataFrame({"col": [1, 2]})
    with patch("pandas.read_csv", return_value=df):
        loaded = load_data("fake_test.csv")
        assert loaded.shape == (2, 1)

def test_evaluate_model():
    # Mock classifier
    clf = MagicMock()
    clf.predict.return_value = np.array([0, 1])
    clf.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
    
    X_test = np.random.rand(2, 5)
    y_test = np.array([0, 1])
    
    metrics = evaluate_model(clf, X_test, y_test)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "auc" in metrics
    assert metrics["accuracy"] == 1.0

def test_save_metrics(tmp_path):
    metrics = {"accuracy": 0.95, "precision": 0.90}
    file_path = tmp_path / "subdir" / "metrics.json"
    save_metrics(metrics, str(file_path))
    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data["accuracy"] == 0.95
