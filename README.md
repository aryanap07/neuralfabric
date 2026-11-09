<p align="center">
  <img
    src="https://raw.githubusercontent.com/aryanap07/NeuralFabric/main/assets/NeuralFabric.svg"
    alt="NeuralFabric"
    width="520"
  >
</p>

<p align="center">
  <strong>A from-scratch machine learning framework built from first principles.</strong>
</p>

NeuralFabric is an open-source machine learning framework that explores the foundations of modern machine learning through hands-on implementation. It starts with a NumPy-based tensor engine and reverse-mode automatic differentiation, then builds toward classical machine learning and deep learning components.

> **Understand the fundamentals. Build the system. Learn by implementing.**

## Features

### Core Functionality

- NumPy-based Tensor Engine
- Reverse-Mode Automatic Differentiation
- Linear Regression
- Binary Logistic Regression
- Train-Test Split
- Feature Scaling with `StandardScaler`
- Regression and Classification Metrics

### Development Tooling

- Static Type Checking with MyPy
- Testing with Pytest
- Linting and Formatting with Ruff
- Continuous Integration with GitHub Actions
- Python Package Build and PyPI Publishing Workflow

## Installation

NeuralFabric is available on PyPI:

```bash
python -m pip install neuralfabric
```

**Requirements:** Python 3.11 or later. The CI workflow tests Python 3.11, 3.12, and 3.13.

### Development Installation

```bash
git clone https://github.com/aryanap07/neuralfabric.git
cd neuralfabric
python -m pip install -e ".[dev]"
pre-commit install
```

## Quick Start

The following example trains a linear regression model on a small dataset:

```python
from neuralfabric.core.tensor import Tensor
from neuralfabric.linear_model import LinearRegression
from neuralfabric.model_selection import train_test_split

X = Tensor([
    [800, 2],
    [1000, 2],
    [1200, 3],
    [1500, 3],
    [1800, 4],
    [2000, 4],
    [2200, 5],
    [2500, 5],
])

y = Tensor([
    [120000],
    [150000],
    [180000],
    [220000],
    [260000],
    [290000],
    [320000],
    [370000],
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

model = LinearRegression(
    lr=1e-8,
    epochs=10000,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:")
print(predictions.data)

print("\nR² Score:")
print(model.score(X_test, y_test))
```

## Implemented Components

| Component Type | Implementation |
| :--- | :--- |
| Regressor | `LinearRegression` |
| Classifier | `LogisticRegression` |
| Transformer | `StandardScaler` |

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── publish.yml
├── assets/
│   └── NeuralFabric.svg
├── examples/
├── src/
│   └── neuralfabric/
│       ├── core/
│       │   └── tensor.py
│       ├── linear_model/
│       │   ├── linear_regression.py
│       │   └── logistic_regression.py
│       ├── metrics/
│       │   ├── classification.py
│       │   └── regression.py
│       ├── model_selection/
│       │   └── train_test_split.py
│       └── preprocessing/
│           └── scalers.py
├── tests/
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── SECURITY.md
```

## Roadmap

### Completed

- [x] Tensor Engine
- [x] Automatic Differentiation
- [x] Linear Regression
- [x] Logistic Regression
- [x] Train-Test Split
- [x] StandardScaler
- [x] Regression and Classification Metrics
- [x] Test Suite
- [x] GitHub Actions CI
- [x] PyPI Publishing Workflow

### Planned

- [ ] Neural Network API and Layers
- [ ] Optimizers and Loss Functions
- [ ] Decision Trees and Ensemble Methods
- [ ] Support Vector Machines
- [ ] Clustering Algorithms
- [ ] Dimensionality Reduction
- [ ] Dataset and DataLoader API
- [ ] Model Serialization
- [ ] Documentation and Expanded Examples
- [ ] Benchmark Suite

## Development

Install the development dependencies and pre-commit hooks:

```bash
make dev
pre-commit install
```

Run the project checks and build the package:

```bash
make format
make lint
make test
make build
```

## Contributing

Contributions are welcome. Before opening a pull request, please read the [Contributing Guide](CONTRIBUTING.md) and make sure the relevant tests and checks pass.

## Security

Please report suspected vulnerabilities privately, following the instructions in [SECURITY.md](SECURITY.md). Do not disclose security vulnerabilities through public issues or pull requests.

## License

NeuralFabric is distributed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.
