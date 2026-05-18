import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from svm_socp_lp_solvers import SVMLp, SOCPLp


@pytest.fixture
def binary_data():
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=5,
        n_redundant=2, random_state=0
    )
    return X, y


@pytest.mark.parametrize("Estimator", [SVMLp, SOCPLp])
def test_fit_predict_basic(Estimator, binary_data):
    X, y = binary_data
    model = Estimator(random_state=0).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (X.shape[0],)
    assert set(np.unique(pred)).issubset(set(np.unique(y)))


@pytest.mark.parametrize("Estimator", [SVMLp, SOCPLp])
def test_reproducibility(Estimator, binary_data):
    X, y = binary_data
    m1 = Estimator(random_state=42).fit(X, y)
    m2 = Estimator(random_state=42).fit(X, y)
    assert np.allclose(m1.coef_, m2.coef_)


@pytest.mark.parametrize("Estimator", [SVMLp, SOCPLp])
def test_predict_proba_shape_and_sum(Estimator, binary_data):
    X, y = binary_data
    model = Estimator(random_state=0).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all() and (proba <= 1).all()


@pytest.mark.parametrize("Estimator", [SVMLp, SOCPLp])
def test_works_with_string_labels(Estimator, binary_data):
    X, y = binary_data
    y_str = np.where(y == 1, "positive", "negative")
    model = Estimator(random_state=0).fit(X, y_str)
    pred = model.predict(X)
    assert set(pred).issubset({"positive", "negative"})


@pytest.mark.parametrize("Estimator", [SVMLp, SOCPLp])
def test_works_in_pipeline(Estimator, binary_data):
    X, y = binary_data
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", Estimator(random_state=0))])
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.6


@pytest.mark.parametrize("Estimator", [SVMLp, SOCPLp])
def test_cross_val_score(Estimator, binary_data):
    X, y = binary_data
    scores = cross_val_score(Estimator(random_state=0), X, y, cv=3)
    assert len(scores) == 3
    assert scores.mean() > 0.5


def test_sparsity_increases_with_smaller_p(binary_data):
    X, y = binary_data
    m_high = SOCPLp(p=0.9, random_state=0).fit(X, y)
    m_low = SOCPLp(p=0.1, random_state=0).fit(X, y)
    assert m_low.n_selected_features_ <= m_high.n_selected_features_