"""
Custom exceptions for Vehicle Price Prediction system
"""


class VehiclePricePredictionError(Exception):
    """Base exception for all prediction errors"""
    pass


class ModelNotFoundError(VehiclePricePredictionError):
    """Raised when model file is not found"""
    pass


class PreprocessorNotFoundError(VehiclePricePredictionError):
    """Raised when preprocessor file is not found"""
    pass


class InvalidInputError(VehiclePricePredictionError):
    """Raised when input data is invalid"""
    pass


class PredictionError(VehiclePricePredictionError):
    """Raised when prediction fails"""
    pass


class DataValidationError(VehiclePricePredictionError):
    """Raised when data validation fails"""
    pass


class FeatureMissingError(VehiclePricePredictionError):
    """Raised when required features are missing"""
    pass


class ModelLoadError(VehiclePricePredictionError):
    """Raised when model loading fails"""
    pass


class ConfigurationError(VehiclePricePredictionError):
    """Raised when configuration is invalid"""
    pass
