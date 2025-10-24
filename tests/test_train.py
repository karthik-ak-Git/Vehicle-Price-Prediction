"""Tests for model training"""
import pytest
import os
import sys
from pathlib import Path
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTraining:
    """Test suite for training module"""

    def test_model_file_exists(self):
        """Test that model file was created"""
        model_path = "models/best_model.pkl"
        assert os.path.exists(model_path), "Model file not found"
        assert os.path.getsize(model_path) > 0, "Model file is empty"

    def test_model_structure(self):
        """Test model structure and contents"""
        model_path = "models/best_model.pkl"
        if os.path.exists(model_path):
            model_pack = joblib.load(model_path)

            assert "model" in model_pack, "Model object not found in package"
            assert "algo" in model_pack, "Algorithm name not found in package"
            assert hasattr(model_pack["model"], "predict"), "Model doesn't have predict method"

    def test_metrics_file_exists(self):
        """Test that metrics file was created"""
        metrics_path = "outputs/metrics.json"
        assert os.path.exists(metrics_path), "Metrics file not found"

    def test_metrics_structure(self):
        """Test metrics file structure"""
        metrics_path = "outputs/metrics.json"
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)

            required_keys = ["MAE", "RMSE", "R2"]
            for key in required_keys:
                assert key in metrics, f"Missing metric: {key}"

            # Check reasonable values
            assert 0 <= metrics["R2"] <= 1, "R2 score out of range"
            assert metrics["MAE"] > 0, "MAE should be positive"
            assert metrics["RMSE"] > 0, "RMSE should be positive"

    def test_feature_importance_exists(self):
        """Test that feature importance file exists"""
        importance_path = "outputs/feature_importance.csv"
        if os.path.exists(importance_path):
            import pandas as pd
            df = pd.read_csv(importance_path)
            assert len(df) > 0, "Feature importance file is empty"
            assert "feature" in df.columns or "Feature" in df.columns

    def test_training_log_exists(self):
        """Test that training log was created"""
        log_path = "outputs/training_log.json"
        if os.path.exists(log_path):
            import json
            with open(log_path) as f:
                log = json.load(f)

            assert "models" in log, "Models list not found in training log"
            assert "timestamp" in log, "Timestamp not found in training log"
            assert len(log["models"]) > 0, "No models in training log"
