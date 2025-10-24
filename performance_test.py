"""
Performance and load testing for Vehicle Price Prediction API

Run with: locust -f performance_test.py --host=http://localhost:8000
"""
from locust import HttpUser, task, between
import random


class VehiclePricePredictionUser(HttpUser):
    """Simulated user for load testing"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Initialize test data"""
        self.car_makes = ["Maruti", "Toyota", "Honda", "Hyundai", "Tata", "Mahindra"]
        self.fuels = ["Petrol", "Diesel", "CNG"]
        self.transmissions = ["Manual", "Automatic"]
    
    @task(10)
    def predict_price(self):
        """Test single prediction endpoint (most common)"""
        car_data = {
            "make": random.choice(self.car_makes),
            "year": random.randint(2015, 2024),
            "fuel": random.choice(self.fuels),
            "transmission": random.choice(self.transmissions),
            "engine_cc": random.randint(800, 2000),
            "km_driven": random.randint(10000, 100000),
            "max_power_bhp": random.uniform(60, 150),
            "mileage_value": random.uniform(12, 25),
            "seats": random.choice([5, 7]),
            "owner": "First"
        }
        
        with self.client.post("/predict", json=car_data, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if "predicted_price" in result and result["predicted_price"] > 0:
                    response.success()
                else:
                    response.failure("Invalid prediction response")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def get_model_info(self):
        """Test model info endpoint"""
        with self.client.get("/model-info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test health check endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def batch_prediction(self):
        """Test batch prediction endpoint"""
        batch_size = random.randint(2, 10)
        batch_data = []
        
        for _ in range(batch_size):
            car_data = {
                "make": random.choice(self.car_makes),
                "year": random.randint(2015, 2024),
                "fuel": random.choice(self.fuels),
                "transmission": random.choice(self.transmissions),
            }
            batch_data.append(car_data)
        
        with self.client.post("/predict-batch", json=batch_data, catch_response=True) as response:
            if response.status_code == 200:
                results = response.json()
                if len(results) == batch_size:
                    response.success()
                else:
                    response.failure(f"Expected {batch_size} results, got {len(results)}")
            else:
                response.failure(f"Got status code {response.status_code}")


class StressTestUser(HttpUser):
    """Heavy load user for stress testing"""
    
    wait_time = between(0.1, 0.5)  # Faster requests for stress test
    
    @task
    def rapid_predictions(self):
        """Rapid-fire predictions"""
        car_data = {
            "make": "Maruti",
            "year": 2019,
            "fuel": "Petrol",
            "transmission": "Manual"
        }
        self.client.post("/predict", json=car_data)
