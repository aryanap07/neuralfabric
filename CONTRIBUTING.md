# Contributing to NeuralFabric

Thank you for your interest in contributing to NeuralFabric.

NeuralFabric is built around a simple idea: understand machine learning by implementing its foundations from scratch. Contributions that improve the codebase, strengthen the framework, improve performance, expand test coverage, or make the project easier to use are welcome.

## Getting Started

Fork the repository, then clone your fork locally:

```bash
git clone https://github.com/<your-username>/neuralfabric.git
cd neuralfabric

make dev
pre-commit install
```

Before making changes, make sure the existing test suite passes:

```bash
make test
```

## Development Workflow

Create a focused branch from `main`:

```bash
git checkout -b feature/my-feature
```

Make your changes, add or update tests, and run the project checks:

```bash
make format
make lint
make test
```

Use clear and descriptive commit messages:

```bash
git commit -m "feat: add decision tree classifier"
```

Push your branch and open a Pull Request against `main`.

Keep Pull Requests focused. Smaller changes are easier to review, test, and maintain.

## Coding Standards

NeuralFabric follows a simple set of engineering principles:

* Follow PEP 8 and existing project conventions.
* Use type annotations throughout the codebase.
* Prefer clear, readable, and maintainable implementations.
* Keep functions and classes focused on a single responsibility.
* Use descriptive names for variables, functions, and classes.
* Avoid unnecessary dependencies.
* Keep public APIs consistent across modules.

When introducing a new abstraction, prefer a simple design that can be extended later rather than adding unnecessary complexity.

## Project Structure

Follow the existing repository structure when adding new code.

```text
src/
└── neuralfabric/
    ├── core/
    ├── linear_model/
    └── model_selection/

tests/
└── ...
```

New modules should be placed in the package that best matches their responsibility.

For example, new linear or logistic models belong under:

```text
src/neuralfabric/linear_model/
```

Tests should mirror the corresponding source structure:

```text
tests/linear_model/
```

Do not introduce new top-level packages unless the change requires a new architectural component.

## Machine Learning APIs

New estimators should follow a consistent interface wherever practical:

```python
model.fit(X, y)
model.predict(X)
```

Additional methods, such as `score`, should follow the conventions established by existing estimators.

Consistency across the API is important because NeuralFabric is intended to grow into a unified framework rather than a collection of unrelated implementations.

## Testing

Every new feature and bug fix should include appropriate tests.

For example:

```text
src/neuralfabric/linear_model/logistic_regression.py
tests/linear_model/test_logistic_regression.py
```

Run the complete test suite with:

```bash
make test
```

Good tests should:

* Verify expected behavior.
* Cover important edge cases.
* Detect regressions.
* Keep implementations honest about their mathematical behavior.

When fixing a bug, add a regression test whenever practical.

## Documentation

Documentation is part of the implementation.

When adding or changing a public feature:

* Update the relevant documentation.
* Add a concise usage example when useful.
* Keep docstrings clear and informative.
* Document behavior that may not be obvious from the API.

Example:

```python
def predict(X):
    """Predict target values for input samples."""
```

Keep documentation accurate and aligned with the current implementation. Avoid documenting planned features as if they already exist.

## Pull Requests

Before opening a Pull Request, verify that:

```bash
make format
make lint
make test
```

complete successfully.

A Pull Request should include:

* A clear title.
* A concise description of the change.
* Tests for new or changed behavior.
* Documentation updates where necessary.
* Any relevant implementation or design notes.

### Pull Request Checklist

* [ ] Code follows project conventions
* [ ] Tests added or updated
* [ ] Documentation updated where necessary
* [ ] Formatting passes
* [ ] Linting passes
* [ ] Tests pass
* [ ] Pull Request is focused and ready for review

## Reporting Issues

Before opening an issue, check whether it has already been reported.

When reporting a bug, include:

* A clear description of the problem.
* Steps to reproduce it.
* Expected behavior.
* Actual behavior.
* Python version.
* Operating system.
* Relevant traceback or logs.
* A minimal reproducible example when possible.

Clear issue reports make problems much easier to reproduce and resolve.

## Feature Requests

Feature requests are welcome.

A useful feature request should explain:

* The problem or use case.
* The proposed functionality.
* Why it would be useful to NeuralFabric.
* Any relevant design considerations or alternatives.

For larger changes, opening an issue before implementation can help establish the direction and avoid duplicated work.

## Code of Conduct

Please keep discussions constructive, respectful, and focused on the project.

Contributors should be open to technical feedback and treat other contributors with professionalism.

## License

By contributing to NeuralFabric, you agree that your contributions will be licensed under the **MIT License**.
