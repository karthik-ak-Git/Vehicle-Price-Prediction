"""
Configuration management for Vehicle Price Prediction
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "Vehicle-Price-Prediction"
    app_version: str = "2.0.0"
    environment: str = "development"
    debug: bool = True
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = True
    
    # Streamlit
    streamlit_port: int = 8501
    streamlit_server_address: str = "0.0.0.0"
    
    # Model paths
    model_path: str = "models/best_model.pkl"
    preprocessor_path: str = "outputs/preprocessor.joblib"
    model_version: str = "2.0.0"
    
    # Data paths
    data_dir: str = "dataset/"
    output_dir: str = "outputs/"
    models_dir: str = "models/"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    cors_origins: str = "http://localhost:3000,http://localhost:8501"
    allowed_hosts: str = "localhost,127.0.0.1"
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = "logs/app.log"
    
    # Monitoring
    metrics_enabled: bool = True
    metrics_port: int = 9090
    
    # Feature flags
    enable_batch_predictions: bool = True
    enable_model_explanations: bool = True
    enable_performance_monitoring: bool = True
    
    # Training
    training_random_state: int = 42
    training_n_iter: int = 25
    training_cv_folds: int = 3
    training_use_gpu: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def cors_origins_list(self) -> list:
        """Parse CORS origins into list"""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def allowed_hosts_list(self) -> list:
        """Parse allowed hosts into list"""
        return [host.strip() for host in self.allowed_hosts.split(",")]


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
