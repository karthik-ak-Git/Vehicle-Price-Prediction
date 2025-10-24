# Quick Setup Script for Windows PowerShell
# Run: .\setup.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Vehicle Price Prediction - Quick Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.9 or higher from https://www.python.org/" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install development dependencies
Write-Host "`nInstalling development dependencies..." -ForegroundColor Yellow
pip install pytest pytest-cov black flake8 isort mypy pre-commit locust

# Create necessary directories
Write-Host "`nCreating project directories..." -ForegroundColor Yellow
$directories = @("models", "outputs", "dataset", "logs")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "✓ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "✓ Exists: $dir" -ForegroundColor Green
    }
}

# Create .env file if it doesn't exist
Write-Host "`nSetting up environment configuration..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✓ Created .env file from template" -ForegroundColor Green
    Write-Host "  Please edit .env file with your configuration" -ForegroundColor Cyan
} else {
    Write-Host "✓ .env file already exists" -ForegroundColor Green
}

# Install pre-commit hooks
Write-Host "`nInstalling pre-commit hooks..." -ForegroundColor Yellow
pre-commit install
Write-Host "✓ Pre-commit hooks installed" -ForegroundColor Green

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Setup Complete! ✓" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Place your data files in the 'dataset/' directory" -ForegroundColor White
Write-Host "2. Process data:   python data/dataloader.py --dataset_dir dataset/ --out outputs/" -ForegroundColor White
Write-Host "3. Train model:    python train.py" -ForegroundColor White
Write-Host "4. Evaluate model: python evaluate.py" -ForegroundColor White
Write-Host "5. Start API:      uvicorn api_app:app --reload" -ForegroundColor White
Write-Host "6. Start UI:       streamlit run streamlit_app.py" -ForegroundColor White
Write-Host ""
Write-Host "Testing:" -ForegroundColor Yellow
Write-Host "  pytest tests/ -v" -ForegroundColor White
Write-Host ""
Write-Host "Documentation:" -ForegroundColor Yellow
Write-Host "  README.md, CONTRIBUTING.md, MODEL_CARD.md" -ForegroundColor White
Write-Host ""
