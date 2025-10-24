"""Pytest configuration and shared fixtures"""
import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_car_data():
    """Sample car data for testing"""
    return {
        "make": "Toyota",
        "year": 2018,
        "fuel": "Petrol",
        "transmission": "Manual",
        "engine_cc": 1200,
        "km_driven": 50000,
        "max_power_bhp": 85.0,
        "mileage_value": 18.0,
        "seats": 5,
        "owner": "First",
        "seller_type": "Individual",
        "torque_nm": 100.0,
        "torque_rpm": 2000.0,
        "mileage_unit": "kmpl"
    }


@pytest.fixture
def sample_car_list():
    """Multiple cars for batch testing"""
    return [
        {"make": "Maruti", "year": 2019, "fuel": "Petrol", "transmission": "Manual", "engine_cc": 1200, "km_driven": 30000},
        {"make": "Honda", "year": 2020, "fuel": "Diesel", "transmission": "Automatic", "engine_cc": 1500, "km_driven": 20000},
        {"make": "Hyundai", "year": 2017, "fuel": "Petrol", "transmission": "Manual", "engine_cc": 1000, "km_driven": 60000}
    ]


@pytest.fixture
def model_paths():
    """Paths to model artifacts"""
    return {
        "model": "models/best_model.pkl",
        "preprocessor": "outputs/preprocessor.joblib",
        "processed_data": "outputs/processed_data.pkl"
    }


@pytest.fixture
def api_base_url():
    """Base URL for API testing"""
    return "http://127.0.0.1:8000"
