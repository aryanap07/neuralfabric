# NeuralFabric

A from-scratch machine learning and deep learning framework, from tensors to transformers.

## Features

- Tensor engine built on NumPy
- Automatic differentiation (Autograd)
- Linear Regression
- Logistic Regression
- Train-test split utility
- Type-safe codebase with MyPy
- Comprehensive test suite with Pytest
- Ruff for code quality
- GitHub Actions CI/CD workflows
- Modern Python packaging and PyPI distribution

## Installation

```bash
pip install neuralfabric
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

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

## Available Models

| Category | Model |
|----------|--------|
| Regression | LinearRegression |
| Classification | LogisticRegression |

## Project Structure

```text
src/neuralfabric/
├── core/
│   └── Tensor and Autograd Engine
├── linear_model/
│   ├── LinearRegression
│   └── LogisticRegression
└── model_selection/
    └── train_test_split
```

## Roadmap

### Completed

- [x] Tensor implementation
- [x] Automatic differentiation engine
- [x] Linear Regression
- [x] Logistic Regression
- [x] Train-test split utility
- [x] Unit testing
- [x] CI/CD workflows
- [x] PyPI publishing

### Planned

- [ ] Decision Trees
- [ ] Random Forests
- [ ] Support Vector Machines
- [ ] K-Means Clustering
- [ ] Principal Component Analysis (PCA)
- [ ] Neural Network API
- [ ] Optimizers (SGD, Adam)
- [ ] Transformer Architecture
- [ ] Dataset utilities
- [ ] Documentation website

## Development

```bash
make dev
make test
make lint
make format
make build
make publish
```

## Contributing

Contributions, bug reports, feature requests, and discussions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

## License

Distributed under the MIT License. See the LICENSE file for details.
