# Contributing to TRACE

Thank you for your interest in contributing! This document provides guidelines
for setting up the development environment, running tests, and submitting
changes.

## Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/IoannisPerachoritis-hub/TRACE.git
   cd TRACE
   ```

2. **Create a virtual environment (Python >= 3.11):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. **Install in editable mode with dev dependencies:**

   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run a specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=gwas --cov-report=term-missing
```

All tests should pass before submitting a pull request.

## Code Style

- Follow existing patterns in the codebase.
- Use type hints for function signatures.
- Keep functions focused and modular.
- Prefer NumPy vectorized operations over Python loops.
- Add docstrings for public functions.

## Reporting Issues

Open an issue on the GitHub repository with:

- A clear description of the problem or feature request.
- Steps to reproduce (for bugs).
- Your Python version and OS.
- Relevant error messages or logs.

## Pull Request Workflow

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, keeping commits focused and well-described.
3. Add or update tests for any new functionality.
4. Ensure `pytest tests/ -v` passes with no failures.
5. Open a pull request against `main` with a clear description of the changes.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License (see [LICENSE](LICENSE)).
