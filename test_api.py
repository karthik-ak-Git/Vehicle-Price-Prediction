#!/usr/bin/env python3
"""
Comprehensive Test Suite for Vehicle Price Prediction API

Tests health checks, predictions, error handling, and edge cases
"""
import requests
import json
import pytest
import time
from typing import Dict, Any


class TestAPI:
    """API test suite using pytest"""

    BASE_URL = "http://127.0.0.1:8000"

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup before each test"""
        # Check if API is running
        try:
            requests.get(f"{self.BASE_URL}/", timeout=2)
        except requests.exceptions.RequestException:
            pytest.skip("API is not running. Start with: uvicorn api_app:app")

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data

    def test_prediction_success(self):
        """Test successful prediction"""
        car_data = {
            "make": "Toyota",
            "year": 2018,
            "fuel": "Petrol",
            "transmission": "Manual",
            "engine_cc": 1200,
            "km_driven": 50000,
            "max_power_bhp": 85.0,
            "owner": "First"
        }

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=car_data
        )

        assert response.status_code == 200
        result = response.json()
        assert "predicted_price" in result
        assert "formatted_price" in result
        assert "model_used" in result
        assert result["predicted_price"] > 0

    def test_prediction_with_defaults(self):
        """Test prediction with minimal data (uses defaults)"""
        car_data = {
            "make": "Maruti",
            "year": 2019
        }

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=car_data
        )

        assert response.status_code == 200
        result = response.json()
        assert result["predicted_price"] > 0

    def test_prediction_invalid_fuel(self):
        """Test validation with invalid fuel type"""
        car_data = {
            "make": "Toyota",
            "year": 2018,
            "fuel": "Nuclear",  # Invalid
            "transmission": "Manual"
        }

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=car_data
        )

        assert response.status_code == 422  # Validation error

    def test_prediction_invalid_year(self):
        """Test validation with invalid year"""
        car_data = {
            "make": "Toyota",
            "year": 1800,  # Too old
            "fuel": "Petrol"
        }

        response = requests.post(
            f"{self.BASE_URL}/predict",
            json=car_data
        )

        assert response.status_code == 422  # Validation error

    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = requests.get(f"{self.BASE_URL}/model-info")

        assert response.status_code == 200
        info = response.json()
        assert "model_name" in info
        assert "status" in info
        assert info["status"] == "ready"

    def test_multiple_predictions(self):
        """Test multiple predictions in sequence"""
        test_cars = [
            {"make": "Maruti", "year": 2019, "fuel": "Petrol"},
            {"make": "Honda", "year": 2020, "fuel": "Diesel"},
            {"make": "Toyota", "year": 2018, "fuel": "Petrol"}
        ]

        for car in test_cars:
            response = requests.post(f"{self.BASE_URL}/predict", json=car)
            assert response.status_code == 200
            assert response.json()["predicted_price"] > 0

    def test_prediction_performance(self):
        """Test prediction response time"""
        car_data = {
            "make": "Toyota",
            "year": 2018,
            "fuel": "Petrol"
        }

        start = time.time()
        response = requests.post(f"{self.BASE_URL}/predict", json=car_data)
        duration = time.time() - start

        assert response.status_code == 200
        assert duration < 2.0  # Should respond within 2 seconds


# Test data
test_car = {
    "make": "Toyota",
    "year": 2018,
    "fuel": "Petrol",
    "transmission": "Manual",
    "engine_cc": 1200,
    "km_driven": 50000,
    "max_power_bhp": 85.0,
    "owner": "First"
}


def test_api():
    """Standalone test function (non-pytest)"""
    print("🧪 Testing Vehicle Price Prediction API")
    print("=" * 50)

    base_url = "http://127.0.0.1:8000"

    # Test health endpoint
    try:
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()['message']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        print("   Make sure the API is running: uvicorn api_app:app --host 127.0.0.1 --port 8000")
        return

    # Test prediction endpoint
    try:
        print("\n2. Testing prediction endpoint...")
        response = requests.post(
            f"{base_url}/predict",
            json=test_car,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful")
            print(f"   Car: {test_car['make']} {test_car['year']}")
            print(f"   Predicted Price: {result['formatted_price']}")
            print(f"   Category: {result['price_category']}")
            print(f"   Model: {result['model_used']}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Prediction request failed: {e}")

    # Test model info endpoint
    try:
        print("\n3. Testing model info endpoint...")
        response = requests.get(f"{base_url}/model-info")
        if response.status_code == 200:
            info = response.json()
            print("✅ Model info retrieved")
            print(f"   Model: {info['model_name']}")
            print(f"   Features: {info['feature_count']}")
            print(f"   Status: {info['status']}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info request failed: {e}")

    print("\n🎉 API testing completed!")


if __name__ == "__main__":
    test_api()
