#!/usr/bin/env python3
"""
Benchmark script for model inference performance
"""
import time
import statistics
from typing import List
import joblib
import pandas as pd
import numpy as np
from datetime import datetime


def load_artifacts():
    """Load model and preprocessor"""
    print("Loading artifacts...")
    model_pack = joblib.load("models/best_model.pkl")
    model = model_pack["model"]
    preprocessor = joblib.load("outputs/preprocessor.joblib")
    return model, preprocessor


def generate_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate random test data"""
    np.random.seed(42)
    
    data = {
        'km_driven': np.random.randint(10000, 150000, n_samples),
        'mileage_value': np.random.uniform(10, 25, n_samples),
        'engine_cc': np.random.randint(800, 2000, n_samples),
        'max_power_bhp': np.random.uniform(60, 150, n_samples),
        'torque_nm': np.random.uniform(80, 250, n_samples),
        'torque_rpm': np.random.uniform(1500, 4000, n_samples),
        'seats': np.random.choice([5, 7], n_samples),
        'age': np.random.randint(1, 10, n_samples),
        'fuel': np.random.choice(['Petrol', 'Diesel'], n_samples),
        'transmission': np.random.choice(['Manual', 'Automatic'], n_samples),
        'owner': np.random.choice(['First', 'Second'], n_samples),
        'seller_type': np.random.choice(['Individual', 'Dealer'], n_samples),
        'mileage_unit': 'kmpl',
        'make': np.random.choice(['Maruti', 'Toyota', 'Honda'], n_samples)
    }
    
    return pd.DataFrame(data)


def benchmark_single_prediction(model, preprocessor, data: pd.DataFrame, n_runs: int = 100) -> List[float]:
    """Benchmark single prediction performance"""
    times = []
    
    for i in range(n_runs):
        sample = data.iloc[[i % len(data)]]
        
        start = time.perf_counter()
        X = preprocessor.transform(sample)
        _ = model.predict(X)
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    return times


def benchmark_batch_prediction(model, preprocessor, data: pd.DataFrame, batch_sizes: List[int]) -> dict:
    """Benchmark batch prediction performance"""
    results = {}
    
    for batch_size in batch_sizes:
        batch = data.iloc[:batch_size]
        
        start = time.perf_counter()
        X = preprocessor.transform(batch)
        _ = model.predict(X)
        end = time.perf_counter()
        
        total_time = (end - start) * 1000
        time_per_sample = total_time / batch_size
        
        results[batch_size] = {
            'total_ms': total_time,
            'per_sample_ms': time_per_sample,
            'throughput': 1000 / time_per_sample  # predictions per second
        }
    
    return results


def print_results(single_times: List[float], batch_results: dict):
    """Print benchmark results"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 Single Prediction Performance")
    print("-" * 60)
    print(f"  Mean:   {statistics.mean(single_times):.2f} ms")
    print(f"  Median: {statistics.median(single_times):.2f} ms")
    print(f"  Min:    {min(single_times):.2f} ms")
    print(f"  Max:    {max(single_times):.2f} ms")
    print(f"  StdDev: {statistics.stdev(single_times):.2f} ms")
    print(f"  P95:    {statistics.quantiles(single_times, n=20)[18]:.2f} ms")
    print(f"  P99:    {statistics.quantiles(single_times, n=100)[98]:.2f} ms")
    
    print("\n📦 Batch Prediction Performance")
    print("-" * 60)
    print(f"{'Batch Size':<12} {'Total (ms)':<12} {'Per Sample':<12} {'Throughput (pred/s)'}")
    print("-" * 60)
    
    for batch_size, metrics in sorted(batch_results.items()):
        print(f"{batch_size:<12} {metrics['total_ms']:<12.2f} {metrics['per_sample_ms']:<12.2f} {metrics['throughput']:<.2f}")
    
    print("\n" + "="*60)
    print("✅ Benchmark completed successfully!")
    print("="*60)


def main():
    print("🚀 Vehicle Price Prediction - Performance Benchmark\n")
    
    # Load artifacts
    model, preprocessor = load_artifacts()
    print("✅ Artifacts loaded\n")
    
    # Generate test data
    print("Generating test data...")
    data = generate_test_data(1000)
    print(f"✅ Generated {len(data)} test samples\n")
    
    # Benchmark single predictions
    print("Running single prediction benchmark...")
    single_times = benchmark_single_prediction(model, preprocessor, data, n_runs=100)
    print("✅ Single prediction benchmark complete\n")
    
    # Benchmark batch predictions
    print("Running batch prediction benchmark...")
    batch_sizes = [1, 10, 50, 100, 500, 1000]
    batch_results = benchmark_batch_prediction(model, preprocessor, data, batch_sizes)
    print("✅ Batch prediction benchmark complete\n")
    
    # Print results
    print_results(single_times, batch_results)


if __name__ == "__main__":
    main()
