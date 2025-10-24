# Contributing to Vehicle Price Prediction 🚗

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## 📜 Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Vehicle-Price-Prediction.git
   cd Vehicle-Price-Prediction
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/karthik-ak-Git/Vehicle-Price-Prediction.git
   ```

## 💻 Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\Activate.ps1  # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install development dependencies**:
   ```bash
   pip install pytest pytest-cov black flake8 isort mypy pre-commit
   ```

4. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Run tests to verify setup**:
   ```bash
   pytest tests/ -v
   ```

## 🤝 How to Contribute

### Reporting Bugs

- Use GitHub Issues
- Include clear title and description
- Provide steps to reproduce
- Include system information (OS, Python version)
- Add relevant logs or screenshots

### Suggesting Enhancements

- Use GitHub Issues with "enhancement" label
- Clearly describe the feature
- Explain why it would be useful
- Consider implementation approach

### Code Contributions

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clear, documented code
   - Follow coding standards
   - Add/update tests
   - Update documentation

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```
   
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `style:` formatting changes
   - `refactor:` code refactoring
   - `test:` adding/updating tests
   - `chore:` maintenance tasks

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request** on GitHub

## 📝 Coding Standards

### Python Style Guide

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Maximum line length: 120 characters
- Use meaningful variable names

### Code Formatting

Run formatters before committing:

```bash
# Format with Black
black .

# Sort imports
isort .

# Check linting
flake8 .

# Type checking
mypy .
```

### Documentation

- Add docstrings to all functions, classes, and modules
- Use Google-style docstrings:
  ```python
  def predict_price(car_data: dict) -> float:
      """
      Predict vehicle price from car features.
      
      Args:
          car_data: Dictionary containing car features
          
      Returns:
          Predicted price in rupees
          
      Raises:
          ValueError: If required features are missing
      """
      pass
  ```

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use pytest fixtures for setup
- Aim for >80% code coverage

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_predict.py

# Run specific test
pytest tests/test_predict.py::TestPredictor::test_basic_prediction
```

### Test Types

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **API Tests**: Test REST API endpoints
4. **Performance Tests**: Test response times and resource usage

## 🔄 Pull Request Process

### Before Submitting

- [ ] Run all tests and ensure they pass
- [ ] Run linters and formatters
- [ ] Update documentation
- [ ] Add/update tests for new features
- [ ] Update CHANGELOG.md
- [ ] Ensure CI/CD pipeline passes

### PR Guidelines

- **Title**: Clear, concise description following conventional commits
- **Description**: 
  - What changes were made
  - Why the changes were necessary
  - How to test the changes
  - Related issues (use "Closes #123")
- **Size**: Keep PRs focused and reasonably sized
- **Reviews**: Address all review comments
- **CI**: Ensure all checks pass

### Review Process

1. Automated checks run on PR
2. Maintainers review code
3. Address feedback and push updates
4. Maintainer approves and merges

## 📦 Release Process

### Version Numbers

Follow [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Creating a Release

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag -a v2.0.0 -m "Version 2.0.0"`
4. Push tag: `git push origin v2.0.0`
5. Create GitHub Release with release notes

## 🎯 Areas for Contribution

### High Priority

- [ ] Additional ML models and hyperparameter tuning
- [ ] More comprehensive test coverage
- [ ] Performance optimizations
- [ ] Enhanced error handling
- [ ] Improved documentation

### Medium Priority

- [ ] Additional data sources integration
- [ ] Real-time model updating
- [ ] A/B testing framework
- [ ] Monitoring dashboards
- [ ] Internationalization

### Low Priority

- [ ] Mobile app integration
- [ ] Browser extensions
- [ ] Additional visualization options
- [ ] Custom theming

## 💡 Questions?

- Open a GitHub Discussion
- Check existing issues
- Read the documentation
- Contact maintainers

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

---

**Happy Coding! 🚀**
