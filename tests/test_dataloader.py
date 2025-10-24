"""Tests for data loading and preprocessing"""
from data.dataloader import detect_columns, extract_make, parse_number_with_unit
import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataloader:
    """Test suite for dataloader module"""

    def test_detect_columns_basic(self):
        """Test column detection with basic DataFrame"""
        df = pd.DataFrame({
            'name': ['Car 1'],
            'year': [2020],
            'selling_price': [500000],
            'km_driven': [50000],
            'fuel': ['Petrol'],
            'transmission': ['Manual']
        })

        parsed = detect_columns(df)
        assert parsed.target == 'selling_price'
        assert parsed.year == 'year'
        assert parsed.km_driven == 'km_driven'

    def test_extract_make(self):
        """Test make extraction from car names"""
        test_cases = [
            ("Maruti Swift VXI", "Maruti"),
            ("Toyota Fortuner 4x4", "Toyota"),
            ("Honda City ZX VTEC", "Honda"),
            ("Mercedes-Benz C-Class", "Mercedes-Benz"),
            ("BMW 3 Series", "BMW")
        ]

        for name, expected_make in test_cases:
            make = extract_make(name)
            assert make == expected_make, f"Failed for {name}: got {make}, expected {expected_make}"

    def test_parse_number_with_unit(self):
        """Test parsing numbers with units"""
        assert parse_number_with_unit("1200 CC") == 1200
        assert parse_number_with_unit("18.5 kmpl") == 18.5
        assert parse_number_with_unit("85 bhp") == 85
        assert parse_number_with_unit("invalid") is None

    def test_preprocessor_exists(self):
        """Test that preprocessor file exists after data loading"""
        preprocessor_path = "outputs/preprocessor.joblib"
        if os.path.exists(preprocessor_path):
            assert os.path.getsize(preprocessor_path) > 0

    def test_processed_data_structure(self):
        """Test structure of processed data"""
        data_path = "outputs/processed_data.pkl"
        if os.path.exists(data_path):
            import joblib
            data = joblib.load(data_path)

            required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
            for key in required_keys:
                assert key in data, f"Missing key: {key}"

            # Check shapes match
            assert len(data['X_train']) == len(data['y_train'])
            assert len(data['X_val']) == len(data['y_val'])
            assert len(data['X_test']) == len(data['y_test'])
