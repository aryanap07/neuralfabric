<p align="center">
  <img
    src="https://raw.githubusercontent.com/aryanap07/NeuralFabric/main/assets/NeuralFabric.svg"
    alt="NeuralFabric"
    width="520"
  >
</p>

**A from-scratch machine learning framework built from first principles.**

NeuralFabric is an open-source machine learning framework that explores the foundations of modern ML by implementing core components from scratch. It starts at the tensor level and builds upward through automatic differentiation, classical machine learning, and eventually deep learning.

> **Understand the fundamentals. Build the system. Learn by implementing.**

## Features

* NumPy-based Tensor Engine
* Automatic Differentiation
* Linear Regression
* Logistic Regression
* Train-Test Split
* Type checking with MyPy
* Testing with Pytest
* Code quality with Ruff
* GitHub Actions CI/CD
* Modern Python packaging
* PyPI distribution

## Installation

Install NeuralFabric from PyPI:

```bash
pip install neuralfabric
```

For development:

```bash
git clone https://github.com/your-username/neuralfabric.git
cd neuralfabric
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

| Category       | Model                |
| -------------- | -------------------- |
| Regression     | `LinearRegression`   |
| Classification | `LogisticRegression` |

## Project Structure

```text
src/
└── neuralfabric/
    ├── core/
    │   ├── tensor.py
    │   └── autograd.py
    ├── linear_model/
    │   ├── linear_regression.py
    │   └── logistic_regression.py
    └── model_selection/
        └── train_test_split.py
```

## Roadmap

### Completed

* [x] Tensor Engine
* [x] Automatic Differentiation
* [x] Linear Regression
* [x] Logistic Regression
* [x] Train-Test Split
* [x] Testing
* [x] CI/CD
* [x] PyPI Publishing

### Planned

* [ ] Neural Network API
* [ ] Optimizers and Loss Functions
* [ ] Metrics and Model Serialization
* [ ] Dataset and DataLoader API
* [ ] Decision Trees and Random Forests
* [ ] K-Means and PCA
* [ ] Documentation and Examples
* [ ] Benchmark Suite

## Development

```bash
make dev
make test
make lint
make format
make build
```

## Contributing

Contributions are welcome.

Whether you want to fix a bug, improve documentation, add tests, or build a new component, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Add or update tests.
5. Commit your changes.
6. Open a pull request.

## License

NeuralFabric is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.
