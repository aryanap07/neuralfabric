# Contributing to NeuralFabric

Thank you for your interest in NeuralFabric.

NeuralFabric is an open-source machine learning framework built from first principles. Contributions that improve the core framework, add well-designed algorithms, strengthen tests, improve performance, or make the project easier to understand and use are welcome.

## Getting Started

Fork the repository and clone your fork:

```bash
git clone https://github.com/<your-username>/neuralfabric.git
cd neuralfabric

make dev
pre-commit install
```

Run the test suite before making changes:

```bash
make test
```

This helps ensure your development environment is set up correctly and the existing code is working as expected.

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

Keep changes focused. Small, well-defined Pull Requests are easier to review, test, and maintain.

## Coding Standards

NeuralFabric aims for code that is simple, readable, and maintainable.

* Follow PEP 8 and existing project conventions.
* Use type annotations throughout the codebase.
* Prefer clear and descriptive names.
* Keep functions and classes focused on a single responsibility.
* Avoid unnecessary dependencies.
* Follow existing API and architectural patterns.
* Keep implementations as simple as the problem allows.

For mathematical or algorithmic code, prefer correctness and clarity over unnecessary abstraction.

## Project Structure

Follow the existing structure when adding new code:

```text
src/
└── neuralfabric/
    ├── core/
    ├── linear_model/
    └── model_selection/

tests/
└── ...
```

Place new functionality in the package that best matches its responsibility.

For example:

```text
src/neuralfabric/linear_model/
tests/linear_model/
```

Avoid introducing new top-level packages unless the change represents a genuine architectural component.

## Machine Learning APIs

New estimators should follow the existing API style where practical:

```python
model.fit(X, y)
model.predict(X)
```

Additional methods, such as `score`, should be consistent with the conventions already established in NeuralFabric.

A consistent API makes the framework easier to learn, use, and extend.

## Testing

Every new feature and bug fix should include appropriate tests.

A typical test layout mirrors the source code:

```text
src/neuralfabric/linear_model/logistic_regression.py
tests/linear_model/test_logistic_regression.py
```

Run the full test suite with:

```bash
make test
```

Good tests should verify expected behavior, cover important edge cases, and help prevent regressions.

When fixing a bug, add a regression test whenever practical.

## Documentation

Documentation is part of the contribution.

When adding or changing a public feature:

* Update the relevant documentation.
* Add a usage example when it improves clarity.
* Keep docstrings concise and meaningful.
* Document behavior that may not be obvious from the API.

For example:

```python
def predict(X):
    """Predict target values for input samples."""
```

Keep documentation aligned with the current implementation. Planned functionality should not be documented as available functionality.

## Pull Requests

Before opening a Pull Request, make sure the following checks pass:

```bash
make format
make lint
make test
```

A good Pull Request should include:

* A clear title.
* A concise explanation of the change.
* Tests for new or modified behavior.
* Documentation updates where necessary.
* Relevant implementation or design notes.

### Pull Request Checklist

* [ ] Code follows project conventions
* [ ] Tests added or updated
* [ ] Documentation updated where necessary
* [ ] Formatting passes
* [ ] Linting passes
* [ ] Tests pass
* [ ] Pull Request is focused and ready for review

## Reporting Issues

Before opening an issue, check whether the problem has already been reported.

For bugs, include:

* A clear description of the problem.
* Steps to reproduce it.
* Expected behavior.
* Actual behavior.
* Python version.
* Operating system.
* Relevant traceback or logs.
* A minimal reproducible example when possible.

Clear issue reports make problems easier to reproduce and resolve.

## Feature Requests

Feature requests are welcome.

Please describe:

* The problem or use case.
* The proposed functionality.
* Why it would be useful to NeuralFabric.
* Any relevant design considerations or alternatives.

For larger changes, discussing the idea in an issue before implementation can help keep the project consistent and avoid duplicated work.

## Code of Conduct

Keep discussions constructive, respectful, and focused on the project.

Technical feedback is part of open-source development. Be open to review and treat other contributors with professionalism.

## License

By contributing to NeuralFabric, you agree that your contributions will be licensed under the **MIT License**.
