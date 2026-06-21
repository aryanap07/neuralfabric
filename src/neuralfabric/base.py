"""BaseEstimator and Mixins (RegressorMixin, ClassifierMixin, TransformerMixin)
shared by every classical estimator in LevelGraphs.
"""


class BaseEstimator:
    """Base class for all estimators.

    Provides ``get_params`` / ``set_params`` (sklearn-style) so every model
    is introspectable and plugs into model_selection utilities (e.g. grid
    search) without extra boilerplate.
    """

    def get_params(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.endswith("_")}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class RegressorMixin:
    """Marks an estimator as a regressor; adds a default R^2 `score`."""

    def score(self, X, y):
        from .metrics.regression import r2_score

        return r2_score(y, self.predict(X))


class ClassifierMixin:
    """Marks an estimator as a classifier; adds a default accuracy `score`."""

    def score(self, X, y):
        from .metrics.classification import accuracy_score

        return accuracy_score(y, self.predict(X))


class TransformerMixin:
    """Adds `fit_transform` for free, given `fit` and `transform`."""

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
