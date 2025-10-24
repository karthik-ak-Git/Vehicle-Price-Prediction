"""Tests for prediction module"""
import pytest
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPredictor:
    """Test suite for prediction module"""

    def test_predictor_initialization(self):
        """Test that predictor can be initialized"""
        if not os.path.exists("models/best_model.pkl"):
            pytest.skip("Model file not found")

        from predict import VehiclePricePredictor
        predictor = VehiclePricePredictor()

        assert predictor.model is not None, "Model not loaded"
        assert predictor.preprocessor is not None, "Preprocessor not loaded"
        assert predictor.model_name is not None, "Model name not set"

    def test_basic_prediction(self, sample_car_data):
        """Test basic prediction with valid data"""
        if not os.path.exists("models/best_model.pkl"):
            pytest.skip("Model file not found")

        from predict import VehiclePricePredictor
        predictor = VehiclePricePredictor()

        result = predictor.predict(sample_car_data)

        assert "predicted_price" in result
        assert "formatted_price" in result
        assert "model_used" in result
        assert result["predicted_price"] is not None
        assert result["predicted_price"] > 0

    def test_prediction_with_minimal_data(self):
        """Test prediction with minimal required data"""
        if not os.path.exists("models/best_model.pkl"):
            pytest.skip("Model file not found")

        from predict import VehiclePricePredictor
        predictor = VehiclePricePredictor()

        minimal_data = {
            "make": "Maruti",
            "year": 2019,
            "fuel": "Petrol",
            "transmission": "Manual"
        }

        result = predictor.predict(minimal_data)
        assert "predicted_price" in result
        assert result["predicted_price"] > 0

    def test_prediction_with_different_makes(self):
        """Test predictions for different car makes"""
        if not os.path.exists("models/best_model.pkl"):
            pytest.skip("Model file not found")

        from predict import VehiclePricePredictor
        predictor = VehiclePricePredictor()

        makes = ["Maruti", "Toyota", "Honda", "Hyundai", "BMW"]

        for make in makes:
            car_data = {
                "make": make,
                "year": 2019,
                "fuel": "Petrol",
                "transmission": "Manual",
                "engine_cc": 1200,
                "km_driven": 50000
            }
            result = predictor.predict(car_data)
            assert result["predicted_price"] > 0, f"Invalid prediction for {make}"

    def test_prediction_price_ranges(self):
        """Test that predictions are in reasonable ranges"""
        if not os.path.exists("models/best_model.pkl"):
            pytest.skip("Model file not found")

        from predict import VehiclePricePredictor
        predictor = VehiclePricePredictor()

        # Budget car
        budget_car = {
            "make": "Maruti",
            "year": 2015,
            "fuel": "Petrol",
            "transmission": "Manual",
            "engine_cc": 800,
            "km_driven": 80000
        }

        result = predictor.predict(budget_car)
        # Budget cars should be under 10L
        assert 50000 < result["predicted_price"] < 1000000

    def test_age_calculation(self):
        """Test that age is calculated correctly from year"""
        if not os.path.exists("models/best_model.pkl"):
            pytest.skip("Model file not found")

        from predict import VehiclePricePredictor
        from datetime import datetime

        predictor = VehiclePricePredictor()
        car_data = {"year": 2020, "make": "Toyota"}

        df = predictor._validate_and_prepare_input(car_data)
        expected_age = datetime.now().year - 2020
        assert df['age'].iloc[0] == expected_age
