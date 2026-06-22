# NeuralFabric

A from-scratch machine learning and deep learning framework, from tensors to transformers.

## Features

* Tensor engine built on NumPy
* Automatic differentiation (Autograd)
* Linear Regression
* Logistic Regression
* Train-test split utility
* Modern Python packaging
* Type-safe codebase with MyPy
* Ruff, Black, Pytest, and GitHub Actions integration

## Installation

```bash
pip install -U neuralfabric
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from neuralfabric.linear_model import LinearRegression
from neuralfabric.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

## Project Structure

```text
src/neuralfabric/
├── core/               # Tensor + autograd engine
├── linear_model/       # Linear Regression, Logistic Regression
├── model_selection/    # train_test_split
└── utils/              # Utility functions
```

## Roadmap

### Completed

* [x] Tensor implementation
* [x] Automatic differentiation engine
* [x] Linear Regression
* [x] Logistic Regression
* [x] Train-test split utility
* [x] Unit testing
* [x] CI/CD workflows
* [x] PyPI publishing

### Upcoming

* [ ] Ridge Regression
* [ ] Lasso Regression
* [ ] Elastic Net
* [ ] Decision Trees
* [ ] Random Forests
* [ ] Support Vector Machines
* [ ] K-Means Clustering
* [ ] PCA
* [ ] Neural Network API
* [ ] Optimizers (SGD, Adam)
* [ ] Transformer Architecture
* [ ] Documentation Website

## Development

```bash
make dev
make test
make lint
make format
make build
make publish
```

## License

MIT License. See LICENSE for details.
