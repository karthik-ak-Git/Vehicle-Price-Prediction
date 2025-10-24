"""
Vehicle Price Prediction Package

A production-ready ML system for predicting vehicle prices.
"""

__version__ = "2.0.0"
__author__ = "Karthik"
__email__ = "karthik@example.com"
__license__ = "MIT"

from .exceptions import (
    VehiclePricePredictionError,
    ModelNotFoundError,
    PreprocessorNotFoundError,
    InvalidInputError,
    PredictionError,
)

__all__ = [
    "VehiclePricePredictionError",
    "ModelNotFoundError",
    "PreprocessorNotFoundError",
    "InvalidInputError",
    "PredictionError",
    "__version__",
]
