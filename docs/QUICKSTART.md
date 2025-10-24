# 🚀 Quick Start Guide

Get up and running with Vehicle Price Prediction in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- Git
- 4GB+ RAM

## Installation (Choose One)

### Option 1: Automated Setup (Recommended)

**Windows PowerShell:**
```powershell
git clone https://github.com/karthik-ak-Git/Vehicle-Price-Prediction.git
cd Vehicle-Price-Prediction
.\setup.ps1
```

**Linux/Mac:**
```bash
git clone https://github.com/karthik-ak-Git/Vehicle-Price-Prediction.git
cd Vehicle-Price-Prediction
chmod +x setup_unix.sh && ./setup_unix.sh
```

### Option 2: Using Makefile

```bash
git clone https://github.com/karthik-ak-Git/Vehicle-Price-Prediction.git
cd Vehicle-Price-Prediction
make install-dev
```

### Option 3: Manual Setup

```bash
# Clone repository
git clone https://github.com/karthik-ak-Git/Vehicle-Price-Prediction.git
cd Vehicle-Price-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt
pip install pytest black flake8 isort mypy  # Dev tools

# Create directories
mkdir -p models outputs dataset logs

# Setup configuration
cp .env.example .env
```

## Quick Usage

### 1. Using Pre-trained Model (If Available)

```bash
# Single prediction (CLI)
python predict.py --make "Toyota" --year 2018 --fuel "Petrol"

# Start API server
uvicorn api_app:app --reload

# Start web dashboard
streamlit run streamlit_app.py
```

### 2. Training Your Own Model

```bash
# Step 1: Add your data to dataset/ folder
# Place CSV files in dataset/ directory

# Step 2: Process data
python data/dataloader.py --dataset_dir dataset/ --out outputs/

# Step 3: Train model
python train.py

# Step 4: Evaluate model
python evaluate.py

# Step 5: Use the model (see above)
```

## Quick Commands (Using Makefile)

```bash
make install        # Install dependencies
make format         # Format code
make lint           # Check code quality
make test           # Run tests
make run-api        # Start API
make run-ui         # Start dashboard
make benchmark      # Performance test
make docker-build   # Build containers
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Open coverage report
# Open htmlcov/index.html in browser
```

## Docker (Quick Deploy)

```bash
# Start all services
docker-compose up --build

# Access services:
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

## API Quick Test

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"make":"Toyota","year":2018,"fuel":"Petrol"}'
```

## Troubleshooting

### Python not found
```bash
# Windows: Install from https://www.python.org/
# Linux: sudo apt install python3.9
# Mac: brew install python@3.9
```

### pip install fails
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Try with --user flag
pip install --user -r requirements.txt
```

### Module not found
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate.ps1  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Port already in use
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn api_app:app --port 8001
```

## Next Steps

1. **Read Documentation**: Check out README.md
2. **Explore API**: Visit http://localhost:8000/docs
3. **Run Tests**: `pytest tests/ -v`
4. **Check Examples**: See notebooks and test files
5. **Contribute**: Read CONTRIBUTING.md

## Need Help?

- 📖 **Full Documentation**: [README.md](README.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/karthik-ak-Git/Vehicle-Price-Prediction/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/karthik-ak-Git/Vehicle-Price-Prediction/discussions)
- 📧 **Email**: karthik@example.com

## Quick Reference

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make test` | Run tests |
| `make run-api` | Start API server |
| `make run-ui` | Start dashboard |
| `make format` | Format code |
| `make lint` | Check code quality |
| `make docker-up` | Start with Docker |
| `python train.py` | Train model |
| `python predict.py` | CLI prediction |

---

**You're all set!** 🎉

Start building amazing vehicle price predictions! 🚗💰
