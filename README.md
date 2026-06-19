# NeuralFabric

**NeuralFabric** is a from-scratch machine learning and deep learning
framework — from linear regression to transformers — built the way
scikit-learn, PyTorch, and TensorFlow are built, not just used.

- Classical models use an **sklearn-style estimator API**
  (`fit` / `predict` / `transform`, `get_params` / `set_params`).
- Neural networks use a **PyTorch-style API**: `Tensor` with autograd,
  `nn.Module`, `optim` optimizers.
- Includes a dedicated `transformer` subpackage (attention, positional
  encoding, encoder/decoder stacks, full Transformer models).

## Installation

```bash
pip install neuralfabric          # once published to PyPI
# or, for local development:
pip install -e ".[dev]"
```

## Quick start (target API — see roadmap)

```python
from neuralfabric.linear_model import LinearRegression
from neuralfabric.model_selection import train_test_split
from neuralfabric.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(r2_score(y_test, preds))
```

## Project structure

```
src/neuralfabric/
├── core/             # Tensor + autograd engine (PyTorch-style backbone)
├── linear_model/      # Linear/Logistic/Ridge/Lasso/ElasticNet
├── tree/               # Decision trees
├── ensemble/           # RandomForest, GradientBoosting, AdaBoost, Bagging
├── svm/                # Support Vector Machines
├── naive_bayes/        # Gaussian / Multinomial Naive Bayes
├── cluster/            # KMeans, DBSCAN, hierarchical clustering
├── decomposition/      # PCA, SVD
├── nn/                 # Module, Parameter, layers, activations, losses
│   └── layers/         # Linear, Conv, Pooling, Normalization, RNN/LSTM/GRU...
├── transformer/         # Attention, positional encoding, encoder/decoder, full models
├── optim/              # SGD, Adam, RMSProp, LR schedulers
├── preprocessing/      # Scalers, encoders, imputers
├── model_selection/    # train_test_split, cross-validation, grid search
├── metrics/            # Regression / classification / clustering metrics
├── datasets/           # Toy dataset loaders
├── utils/              # Validation + math helpers
├── base.py             # BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin
└── pipeline.py          # Pipeline / FeatureUnion
```

## Roadmap

- [ ] `core`: Tensor + autograd engine
- [ ] `linear_model`: Linear & Logistic Regression
- [ ] `tree` / `ensemble`: Decision Tree, Random Forest, Gradient Boosting
- [ ] `svm`, `naive_bayes`, `cluster`, `decomposition`
- [ ] `nn`: Module system, core layers, losses, activations
- [ ] `optim`: SGD, Adam
- [ ] CNN layers (`nn/layers/conv.py`, `pooling.py`)
- [ ] RNN/LSTM/GRU (`nn/layers/recurrent.py`)
- [ ] `transformer`: attention → encoder/decoder → full model
- [ ] Docs site + tutorials
- [ ] First PyPI release (`v0.1.0`)

## Development

```bash
make dev      # install with dev dependencies
make test     # run tests with coverage
make lint     # ruff + mypy
make format   # black
make build    # build sdist + wheel
make publish  # twine upload (requires PyPI credentials)
```

## License

MIT — see [LICENSE](LICENSE).
