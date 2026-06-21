# Contributing to NeuralFabric

Thank you for your interest in contributing to NeuralFabric!

## Getting Started

Clone the repository and install development dependencies:

```bash
git clone https://github.com/aryanap07/neuralfabric.git
cd neuralfabric

make dev
pre-commit install
```

## Development Workflow

1. Create a new branch from `main`.
2. Implement your changes.
3. Add or update tests.
4. Run quality checks:

```bash
make format
make lint
make test
```

5. Commit your changes with a clear commit message.
6. Open a Pull Request describing your contribution.

## Code Style

* Follow PEP 8.
* Use type annotations where appropriate.
* Keep functions and classes focused and well documented.
* Ensure all tests pass before submitting a pull request.

## Project Structure

### Machine Learning Models

Place new estimators in the appropriate package:

```text
linear_model/    # Linear & Logistic Regression
tree/            # Decision Trees
ensemble/        # Random Forests, Boosting
svm/             # Support Vector Machines
cluster/         # Clustering algorithms
```

All estimators should follow a consistent API:

```python
model.fit(X, y)
model.predict(X)
```

### Neural Network Components

Place neural network modules under:

```text
nn/
└── layers/
```

and build upon:

```python
Tensor
Module
Parameter
```

### Transformer Components

Place transformer-related implementations under:

```text
transformer/
```

## Testing

Every new feature should include tests.

Example:

```text
src/neuralfabric/linear_model/logistic_regression.py
tests/linear_model/test_logistic_regression.py
```

Run tests with:

```bash
make test
```

## Reporting Issues

When opening an issue, please include:

* A clear description of the problem
* Steps to reproduce
* Expected behavior
* Environment information (Python version, OS)

## License

By contributing to NeuralFabric, you agree that your contributions will be licensed under the MIT License.
