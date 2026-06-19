# Contributing to NeuralFabric

Thanks for considering a contribution!

## Setup
```bash
git clone https://github.com/aryanap07/neuralfabric.git
make dev          # pip install -e ".[dev]"
pre-commit install
```

## Workflow
1. Create a branch off `main`.
2. Implement your change with tests under `tests/<subpackage>/`.
3. Run `make lint` and `make test` before opening a PR.
4. Keep estimators consistent with the conventions in `src/neuralfabric/base.py`
   (fit/predict/transform, get_params/set_params).

## Adding a new model
- Classical estimator → place it under the matching subpackage
  (e.g. `linear_model/`, `tree/`, `svm/`) and inherit from `BaseEstimator`
  plus the relevant mixin (`RegressorMixin` / `ClassifierMixin`).
- Neural network component → place it under `nn/` (layers in `nn/layers/`)
  and build on `nn.module.Module` / `core.tensor.Tensor`.
- Transformer component → place it under `transformer/`.
- Add a matching test file under `tests/<subpackage>/`.
