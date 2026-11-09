import numpy as np
import pytest

from neuralfabric.core.tensor import Tensor
from neuralfabric.preprocessing import StandardScaler


@pytest.fixture
def X() -> Tensor:
    return Tensor(
        [
            [1.0, 10.0, 5.0],
            [2.0, 20.0, 5.0],
            [3.0, 30.0, 5.0],
            [4.0, 40.0, 5.0],
        ]
    )


class TestStandardScaler:
    def test_output_has_zero_mean_and_unit_std(self, rng):
        X = Tensor(rng.normal(loc=50.0, scale=8.0, size=(200, 4)))

        out = StandardScaler().fit_transform(X)

        np.testing.assert_allclose(out.data.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(out.data.std(axis=0), 1.0, atol=1e-5)

    def test_fit_learns_mean_and_scale(self, X):
        scaler = StandardScaler().fit(X)

        assert scaler.mean_ is not None
        assert scaler.scale_ is not None
        np.testing.assert_allclose(scaler.mean_, [2.5, 25.0, 5.0])
        np.testing.assert_allclose(scaler.scale_[:2], X.data.std(axis=0)[:2])

    def test_constant_feature_is_centered_not_divided_by_zero(self, X):
        out = StandardScaler().fit_transform(X)

        assert np.all(np.isfinite(out.data))
        np.testing.assert_allclose(out.data[:, 2], 0.0)

    def test_inverse_transform_round_trips(self, X):
        scaler = StandardScaler().fit(X)

        restored = scaler.inverse_transform(scaler.transform(X))

        np.testing.assert_allclose(restored.data, X.data, atol=1e-5)

    def test_with_mean_false_skips_centering(self, X):
        scaler = StandardScaler(with_mean=False).fit(X)
        assert scaler.scale_ is not None

        out = scaler.transform(X)

        np.testing.assert_allclose(out.data[:, 0], X.data[:, 0] / scaler.scale_[0])

    def test_with_std_false_skips_scaling(self, X):
        out = StandardScaler(with_std=False).fit_transform(X)

        np.testing.assert_allclose(out.data[:, 1], X.data[:, 1] - 25.0)

    def test_transform_uses_statistics_from_fit(self, X):
        scaler = StandardScaler().fit(X)

        out = scaler.transform(Tensor([[2.5, 25.0, 5.0]]))

        np.testing.assert_allclose(out.data, 0.0, atol=1e-6)

    def test_integer_input_is_supported(self):
        out = StandardScaler().fit_transform(Tensor([[1, 2], [3, 4], [5, 6]]))

        assert np.issubdtype(out.dtype, np.floating)

    def test_transform_before_fit_raises(self, X):
        with pytest.raises(RuntimeError, match="fitted"):
            StandardScaler().transform(X)

    def test_inverse_transform_before_fit_raises(self, X):
        with pytest.raises(RuntimeError, match="fitted"):
            StandardScaler().inverse_transform(X)

    def test_non_2d_input_raises(self):
        with pytest.raises(ValueError, match="2D"):
            StandardScaler().fit(Tensor([1.0, 2.0, 3.0]))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="at least one sample"):
            StandardScaler().fit(Tensor(np.empty((0, 3))))

    def test_feature_count_mismatch_raises(self, X):
        scaler = StandardScaler().fit(X)

        with pytest.raises(ValueError, match="features"):
            scaler.transform(Tensor([[1.0, 2.0]]))

    def test_get_params_excludes_fitted_attributes(self, X):
        scaler = StandardScaler(with_std=False).fit(X)

        assert scaler.get_params() == {"with_mean": True, "with_std": False}

    def test_repr(self):
        assert repr(StandardScaler()) == "StandardScaler(with_mean=True, with_std=True)"
