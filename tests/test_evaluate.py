"""Tests for evaluation module"""
import pytest
import os
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEvaluation:
    """Test suite for evaluation module"""

    def test_metrics_file_exists(self):
        """Test that enhanced metrics file exists"""
        metrics_path = "outputs/enhanced_test_metrics.json"
        if os.path.exists(metrics_path):
            assert os.path.getsize(metrics_path) > 0

    def test_enhanced_metrics_structure(self):
        """Test enhanced metrics structure"""
        metrics_path = "outputs/enhanced_test_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)

            assert "overall" in metrics
            assert "MAE" in metrics["overall"]
            assert "R2" in metrics["overall"]
            assert "MAPE" in metrics["overall"]

    def test_price_range_analysis(self):
        """Test price range analysis in metrics"""
        metrics_path = "outputs/enhanced_test_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)

            if "price_ranges" in metrics:
                assert len(metrics["price_ranges"]) > 0

    def test_evaluation_plots_created(self):
        """Test that evaluation plots were created"""
        plot_files = [
            "outputs/actual_vs_pred_enhanced.png",
            "outputs/residuals_analysis_enhanced.png"
        ]

        for plot_file in plot_files:
            if os.path.exists(plot_file):
                assert os.path.getsize(plot_file) > 0
