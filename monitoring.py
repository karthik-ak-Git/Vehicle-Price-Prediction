"""
Monitoring and metrics collection for Vehicle Price Prediction
"""
import time
from functools import wraps
from typing import Callable, Any
import logging

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logging.warning("prometheus_client not installed. Metrics disabled.")


# Metrics definitions
if HAS_PROMETHEUS:
    # Request counters
    prediction_requests_total = Counter(
        'prediction_requests_total',
        'Total number of prediction requests',
        ['status', 'endpoint']
    )
    
    # Prediction latency
    prediction_duration_seconds = Histogram(
        'prediction_duration_seconds',
        'Prediction request duration in seconds',
        ['endpoint']
    )
    
    # Model info
    model_info = Gauge(
        'model_info',
        'Model information',
        ['model_name', 'version']
    )
    
    # Prediction price distribution
    prediction_price_histogram = Histogram(
        'prediction_price_rupees',
        'Distribution of predicted prices',
        buckets=[100000, 300000, 500000, 800000, 1000000, 1500000, 2000000, 5000000, 10000000]
    )
    
    # Error counter
    prediction_errors_total = Counter(
        'prediction_errors_total',
        'Total number of prediction errors',
        ['error_type']
    )
    
    # Active requests
    active_requests = Gauge(
        'active_requests',
        'Number of active prediction requests'
    )


def track_prediction_time(func: Callable) -> Callable:
    """Decorator to track prediction execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not HAS_PROMETHEUS:
            return func(*args, **kwargs)
        
        start_time = time.time()
        active_requests.inc()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Record metrics
            prediction_duration_seconds.labels(endpoint=func.__name__).observe(duration)
            prediction_requests_total.labels(status='success', endpoint=func.__name__).inc()
            
            # Track price if available
            if isinstance(result, dict) and 'predicted_price' in result:
                prediction_price_histogram.observe(result['predicted_price'])
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            prediction_duration_seconds.labels(endpoint=func.__name__).observe(duration)
            prediction_requests_total.labels(status='error', endpoint=func.__name__).inc()
            prediction_errors_total.labels(error_type=type(e).__name__).inc()
            raise
            
        finally:
            active_requests.dec()
    
    return wrapper


def get_metrics() -> str:
    """Get current metrics in Prometheus format"""
    if HAS_PROMETHEUS:
        return generate_latest(REGISTRY)
    return "Metrics not available (prometheus_client not installed)"


def set_model_info(model_name: str, version: str) -> None:
    """Set model information metric"""
    if HAS_PROMETHEUS:
        model_info.labels(model_name=model_name, version=version).set(1)
