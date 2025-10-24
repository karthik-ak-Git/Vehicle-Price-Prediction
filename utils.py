"""
Enhanced API utilities for validation and processing
"""
from typing import Dict, Any, List
import re
from datetime import datetime

from exceptions import InvalidInputError, FeatureMissingError


def validate_car_data(car_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate car data and return list of errors
    
    Args:
        car_data: Dictionary containing car features
        
    Returns:
        Dictionary of validation errors (empty if valid)
    """
    errors = {}
    
    # Year validation
    if 'year' in car_data:
        year = car_data['year']
        current_year = datetime.now().year
        if not isinstance(year, int) or year < 1990 or year > current_year + 1:
            errors['year'] = f"Year must be between 1990 and {current_year + 1}"
    
    # Numeric range validations
    numeric_validations = {
        'km_driven': (0, 1000000, "Kilometers driven"),
        'engine_cc': (50, 8000, "Engine CC"),
        'max_power_bhp': (10, 1000, "Max power BHP"),
        'mileage_value': (5, 50, "Mileage"),
        'seats': (2, 10, "Seats"),
        'torque_nm': (50, 1000, "Torque NM"),
        'torque_rpm': (1000, 8000, "Torque RPM")
    }
    
    for field, (min_val, max_val, label) in numeric_validations.items():
        if field in car_data and car_data[field] is not None:
            value = car_data[field]
            try:
                value = float(value)
                if value < min_val or value > max_val:
                    errors[field] = f"{label} must be between {min_val} and {max_val}"
            except (TypeError, ValueError):
                errors[field] = f"{label} must be a number"
    
    # Categorical validations
    valid_fuels = ['Petrol', 'Diesel', 'CNG', 'Electric', 'Hybrid', 'LPG']
    if 'fuel' in car_data and car_data['fuel'] not in valid_fuels:
        errors['fuel'] = f"Fuel must be one of: {', '.join(valid_fuels)}"
    
    valid_transmissions = ['Manual', 'Automatic']
    if 'transmission' in car_data and car_data['transmission'] not in valid_transmissions:
        errors['transmission'] = f"Transmission must be one of: {', '.join(valid_transmissions)}"
    
    valid_owners = ['First', 'Second', 'Third', 'Fourth & Above', 'Test Drive Car']
    if 'owner' in car_data and car_data['owner'] not in valid_owners:
        errors['owner'] = f"Owner must be one of: {', '.join(valid_owners)}"
    
    return errors


def sanitize_input(text: str) -> str:
    """
    Sanitize text input to prevent injection attacks
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove any potentially dangerous characters
    text = re.sub(r'[<>\"\'%;()&+]', '', text)
    
    # Limit length
    text = text[:200]
    
    return text.strip()


def calculate_prediction_confidence(
    price: float,
    price_range_metrics: Dict[str, Any] = None
) -> tuple[str, float]:
    """
    Calculate confidence level and score for prediction
    
    Args:
        price: Predicted price
        price_range_metrics: Historical metrics by price range
        
    Returns:
        Tuple of (confidence_level, confidence_score)
    """
    # Default confidence based on price range
    if price < 500000:
        # Budget cars - high confidence (more training data)
        confidence_level = "High confidence"
        confidence_score = 0.92
    elif price < 1000000:
        # Mid-range - high confidence
        confidence_level = "High confidence"
        confidence_score = 0.90
    elif price < 2000000:
        # Premium - medium confidence
        confidence_level = "Medium confidence"
        confidence_score = 0.85
    else:
        # Luxury - lower confidence (less training data)
        confidence_level = "Lower confidence"
        confidence_score = 0.78
    
    return confidence_level, confidence_score


def format_indian_currency(amount: float) -> str:
    """
    Format amount in Indian currency format
    
    Args:
        amount: Amount to format
        
    Returns:
        Formatted currency string
    """
    s = f"{amount:.0f}"
    s = s[::-1]
    
    # Indian grouping: 3 digits, then groups of 2
    result = []
    result.append(s[:3])
    s = s[3:]
    
    while s:
        result.append(s[:2])
        s = s[2:]
    
    formatted = ','.join(result)
    return f"₹{formatted[::-1]}"


def extract_feature_importance(
    feature_names: List[str],
    feature_values: List[float],
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Extract top N important features with their contributions
    
    Args:
        feature_names: List of feature names
        feature_values: List of feature importance values
        top_n: Number of top features to return
        
    Returns:
        List of dictionaries with feature importance info
    """
    # Combine and sort
    features = list(zip(feature_names, feature_values))
    features.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Get top N
    top_features = features[:top_n]
    
    # Format results
    result = []
    for name, value in top_features:
        result.append({
            "feature": name,
            "importance": float(value),
            "percentage": float(value * 100) if value < 1 else float(value)
        })
    
    return result
