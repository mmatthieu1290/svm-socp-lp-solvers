# svm-socp-lp-solvers

Sparse Lp-regularized SVM and distributionally robust SOCP classifiers in Python.

[![tests](https://github.com/mmatthieu1290/svm-socp-lp-solvers/actions/workflows/tests.yml/badge.svg)](https://github.com/mmatthieu1290/svm-socp-lp-solvers/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/)

## Overview

`svm-socp-lp-solvers` implements two binary classifiers that produce **sparse coefficient vectors** while retaining competitive classification accuracy. Both rely on minimizing a non-convex Lp quasi-norm penalty (with `0 < p < 1`) via an Iteratively Reweighted L1 (IRL1) scheme:

- **`SVMLp`** — sparse Lp-regularized Support Vector Machine.
- **`SOCPLp`** — distributionally robust variant formulated as a Second-Order Cone Program with Chebyshev-based chance constraints, requiring only class-conditional means and covariances.

The API follows scikit-learn conventions (`fit`, `predict`, `predict_proba`, `coef_`, `intercept_`) and both estimators pass `sklearn.utils.estimator_checks.check_estimator`.

The method is described in Carrasco, M., Iborra, B., Lopez, J., Marechal, M., & Ramos, A.M. (2026), "Sparse Feature Selection via Lp-Quasi-Norm Second-Order Cone Programming", *Pattern Recognition 114043*. DOI: [10.1016/j.patcog.2026.114043](https://doi.org/10.1016/j.patcog.2026.114043).

## Installation

```bash
pip install git+https://github.com/mmatthieu1290/svm-socp-lp-solvers.git
```

Requires Python ≥ 3.10. Dependencies (`numpy`, `scikit-learn` ≥ 1.6, `cvxpy`, `ecos`) are installed automatically.

## Quick start

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from svm_socp_lp_solvers import SOCPLp

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Standardization is recommended — the SOCP constraints are scale-sensitive.
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SOCPLp(p=0.8,alpha_1=0.8,alpha_2=0.7, random_state=0)),
])
pipe.fit(X_train, y_train)

print(f"Test accuracy: {pipe.score(X_test, y_test):.3f}")

clf = pipe.named_steps["clf"]
print(f"Selected features: {clf.n_selected_features_} / {X_train.shape[1]}")
```

The robust variant `SOCPLp` shown above can be replaced by `SVMLp` for the standard sparse Lp-SVM, with the same API.

## Features

- Two scikit-learn–compatible binary classifiers: `SVMLp` and `SOCPLp`.
- Non-convex Lp quasi-norm penalty (`0 < p < 1`) for aggressive feature selection.
- Robust SOCP variant using multivariate Chebyshev chance constraints (no distributional assumptions beyond mean and covariance).
- Reproducible results via `random_state`.
- Iteratively Reweighted L1 scheme; convex subproblems solved with ECOS through CVXPY.
- Diagnostic attributes: `n_selected_features_`, `selected_feature_names_`, `n_non_zeros_coef_per_iteration_`.

## Scikit-learn compatibility

Both `SVMLp` and `SOCPLp` pass `sklearn.utils.estimator_checks.check_estimator` on scikit-learn ≥ 1.6 (56 checks, including `check_classifiers_train`, `check_estimators_pickle`, `check_fit_idempotent`, and `check_n_features_in_after_fitting`).

They integrate with `Pipeline`, `GridSearchCV`, `cross_val_score`, and other scikit-learn utilities.

The estimators are binary-only by design; multiclass tasks must be wrapped externally (e.g., `OneVsRestClassifier`).

## API summary

Both estimators share the following hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `p` | `0.5` | Sparsity exponent. Must satisfy `0 < p < 1`. Smaller values yield sparser solutions. |
| `C` | `1e4` | Penalty weight on slack variables. Must be > 0. |
| `eps` | `1e-5` | Smoothing parameter for the Lp term, preventing singularities at `w_j = 0`. |
| `tol` | `1e-3` | Stopping tolerance on `‖w_{k+1} − w_k‖_∞`. |
| `max_iter` | `100` | Maximum outer (IRL1) iterations. |
| `tol_select_features` | `1e-5` | Threshold above which a coefficient is considered selected. |
| `random_state` | `None` | Controls the random initialization of `w` for reproducibility. |

`SOCPLp` additionally accepts:

| Parameter | Default | Description |
|---|---|---|
| `alpha_1` | `0.5` | Worst-case probability of correctly classifying the positive class. Must satisfy `0 < alpha_1 < 1`. |
| `alpha_2` | `0.5` | Same, for the negative class. |
| `tau` | `None` | Optional. If set, activates two additional linear constraints on the decision function. |

## Attributes after fit

| Attribute | Description |
|---|---|
| `coef_` | Estimated weight vector, shape `(n_features,)`. |
| `intercept_` | Estimated intercept (scalar). |
| `classes_` | Unique class labels observed during `fit`. |
| `n_iter_` | Number of outer iterations actually performed. |
| `n_features_in_` | Number of features seen during `fit`. |
| `feature_names_in_` | Feature names (if `X` was a DataFrame). |
| `n_selected_features_` | Number of coefficients with `|w_j| > tol_select_features`. |
| `selected_feature_names_` | Names of selected features (if available). |
| `n_non_zeros_coef_per_iteration_` | Non-zero coefficient counts per outer iteration. |

## Examples

### Comparing sparsity levels

Smaller values of `p` produce more aggressive feature selection. The following example demonstrates this on a high-dimensional dataset:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from svm_socp_lp_solvers import SOCPLp

X, y = make_classification(
    n_samples=200, n_features=50, n_informative=8,
    n_redundant=5, random_state=0,
)
X = StandardScaler().fit_transform(X)

for p in [0.9, 0.5, 0.1]:
    model = SOCPLp(p=p, random_state=0).fit(X, y)
    n_selected = model.n_selected_features_
    print(f"p={p}: {n_selected}/{X.shape[1]} features selected")
```

### Use with GridSearchCV

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from svm_socp_lp_solvers import SOCPLp

X, y = load_breast_cancer(return_X_y=True)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SOCPLp(random_state=0)),
])

param_grid = {
    "clf__p": [0.1, 0.3, 0.5, 0.7],
    "clf__C": [0.1, 1.0, 10.0],
}

grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=1)
grid.fit(X, y)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.3f}")
```

## Mathematical background

### SVMLp

Solves

$$
\min_{w, b, \xi} \; \sum_{j=1}^{n} (|w_j| + \varepsilon)^{p} + C \sum_{i=1}^{m} \xi_i
\quad \text{s.t.} \quad
y_i (w^\top x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0,
$$

where $x_i \in \mathbb{R}^n$ is the feature vector of observation $i$ and $y_i \in \{-1, +1\}$ its label. The smoothing parameter $\varepsilon > 0$ makes the objective locally Lipschitz and avoids singularities at $w_j = 0$.

### SOCPLp

Solves

$$
\min_{w, b, \xi} \; \sum_{j=1}^{n} (|w_j| + \varepsilon)^{p} + C \sum_{i=1}^{2} \xi_i
$$

subject to

$$
\begin{aligned}
& w^\top \mu_1 + b \ge 1 - \xi_1 + \kappa(\alpha_1) \, \|S_1^\top w\|, \\
& -(w^\top \mu_2 + b) \ge 1 - \xi_2 + \kappa(\alpha_2) \, \|S_2^\top w\|, \\
& \xi \ge 0,
\end{aligned}
$$

where $\kappa(\alpha) = \sqrt{\alpha / (1 - \alpha)}$, $\mu_j$ is the class-$j$ mean, and $S_j$ satisfies $\Sigma_j = S_j S_j^\top$ with $\Sigma_j$ the class-$j$ covariance matrix.

These constraints are a deterministic reformulation, via the multivariate Chebyshev inequality, of the distributionally robust chance constraints

$$
\inf_{\tilde{x}_j \sim (\mu_j, \Sigma_j)} \Pr\left( (-1)^{j+1}(w^\top \tilde{x}_j + b) \ge 0 \right) \ge \alpha_j, \quad j = 1, 2,
$$

where the infimum runs over all distributions sharing the given mean and covariance.

Because $p < 1$, the objective is non-convex. It is minimized by a Majorization–Minimization scheme that solves an Iteratively Reweighted L1 problem (a QP for `SVMLp`, an SOCP for `SOCPLp`) at each iteration.

For full details, see Carrasco, Lopez & Marechal (2026).

## Citation

If you use this package in your research, please cite:

> Carrasco, M., Ibarra, B, Lopez, J., Marechal, M., & Ramos, A.M. (2026). *Sparse Feature Selection via Lp-Quasi-Norm Second-Order Cone Programming*. Pattern Recognition 114043.

BibTeX:

```bibtex
@article{carrasco2026sparse,
  author  = {Carrasco, Miguel and Ibarra, Benjamin and Lopez, Julio and Marechal, Matthieu and Ramos, Angel M.},
  title   = {Sparse Feature Selection via {Lp}-Quasi-Norm Second-Order Cone Programming},
  journal = {Pattern Recognition},
  year    = {2026},
  doi     = {10.1016/j.patcog.2026.114043}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.