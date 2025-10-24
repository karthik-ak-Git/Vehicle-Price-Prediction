#!/bin/bash
# Quick Setup Script for Linux/Mac
# Run: chmod +x setup_unix.sh && ./setup_unix.sh

echo "========================================"
echo "Vehicle Price Prediction - Quick Setup"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check Python installation
echo -e "${YELLOW}Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed${NC}"
    echo -e "${RED}Please install Python 3.9 or higher${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
python -m pip install --upgrade pip

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Install development dependencies
echo -e "\n${YELLOW}Installing development dependencies...${NC}"
pip install pytest pytest-cov black flake8 isort mypy pre-commit locust

# Create necessary directories
echo -e "\n${YELLOW}Creating project directories...${NC}"
for dir in models outputs dataset logs; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}✓ Created: $dir${NC}"
    else
        echo -e "${GREEN}✓ Exists: $dir${NC}"
    fi
done

# Create .env file if it doesn't exist
echo -e "\n${YELLOW}Setting up environment configuration...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file from template${NC}"
    echo -e "${CYAN}  Please edit .env file with your configuration${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Install pre-commit hooks
echo -e "\n${YELLOW}Installing pre-commit hooks...${NC}"
pre-commit install
echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"

# Summary
echo -e "\n========================================"
echo -e "${GREEN}Setup Complete! ✓${NC}"
echo -e "========================================"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "${NC}1. Place your data files in the 'dataset/' directory${NC}"
echo -e "${NC}2. Process data:   python data/dataloader.py --dataset_dir dataset/ --out outputs/${NC}"
echo -e "${NC}3. Train model:    python train.py${NC}"
echo -e "${NC}4. Evaluate model: python evaluate.py${NC}"
echo -e "${NC}5. Start API:      uvicorn api_app:app --reload${NC}"
echo -e "${NC}6. Start UI:       streamlit run streamlit_app.py${NC}"
echo ""
echo -e "${YELLOW}Testing:${NC}"
echo -e "${NC}  pytest tests/ -v${NC}"
echo ""
echo -e "${YELLOW}Documentation:${NC}"
echo -e "${NC}  README.md, CONTRIBUTING.md, MODEL_CARD.md${NC}"
echo ""
