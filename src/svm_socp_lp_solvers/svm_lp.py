import numpy as np
import numpy.linalg as npl
import cvxpy as cp
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, ClassifierMixin
from .utils import prediction_from_w_b,prediction_probas_from_w_b
from sklearn.utils._param_validation import Interval
from numbers import Real, Integral
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data


class SVMLp(BaseEstimator, ClassifierMixin):

    _parameter_constraints = {
        "p": [Interval(Real, 0, 1, closed="neither")],
        "C": [Interval(Real, 0, None, closed="neither")],
        "tau": [Interval(Real, 0, None, closed="neither"), None],
        "eps": [Interval(Real, 0, None, closed="neither")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol_select_features": [Interval(Real, 0, None, closed="neither")],
        "random_state": ["random_state"],
    }

    r"""
    Smoothed sparse Lp-SVM classifier.

    This estimator solves the following optimization problem:

    .. math::

        \min_{w,b,\xi}\ \sum_{j=1}^n (|w_j|+\varepsilon)^p \;+\; C\sum_{i=1}^m \xi_i
        \quad \mathrm{s.t.}\quad
        y_i (w^\top x_i + b) \geq 1 - \xi_i,\ \xi_i \geq 0,\ i=1,\dots,m.

    The smoothing parameter :math:`\varepsilon>0` makes the objective locally
    Lipschitz and avoids singular behavior at :math:`w_j=0`.

    Parameters
    ----------
    p : float, default=0.5
        Exponent controlling sparsity. Must satisfy 0 < p < 1.

    C : float, default=1e4
        Slack penalty parameter. Must be > 0.

    eps : float, default=1e-5
        Smoothing/approximation parameter :math:`\varepsilon>0` used in

    tol : float, default=1e-4
        Tolerance for stopping criteria.         
        
    max_iter : int, default=100
        Maximum iterations for converging

    tol_select_features: float, default=1e-5
        Minimum value for coeficients to select corresponding feature. 
        Warning: if model has been fitted, changing value of tol_select_features changes the attributes
        n_selected_features_ and selected_feature_names_.   


    Methods
    -------
    fit(X, y)
        Fit the model on labeled data.

    predict(X)
        Predict class labels for samples in X.

    predict_proba(X)
        Estimate probability of the positive class.


    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during fit.

    coef_ : ndarray of shape (n_features,)
        Estimated weight vector.

    intercept_ : float
        Estimated intercept.

    n_iter_ : int
        Number of iterations run.

    n_features_in_ : int
        Number of detected features after calling fit()

    feature_names_in_ : ndarray of shape (n_classes,)
            Names of features seen during :term:`fit`. Defined only when `X` has feature names that are all strings.

    n_selected_features_ : int
        Number of selected features after calling fit()

    selected_feature_names_ : ndarray
       Name of selected features seen during :term:`fit`. Defined only when `X` has feature names that are all strings. 

    n_non_zeros_coef_per_iteration_ : ndarray
       Number of nonzeros componentes of coef_ at each step from step 1.
                 

    Notes
    -----
    The problem is nonconvex given that p < 1; the solver may converge to a local
    minimum depending on the parameters.

    Example 
    -----

    from svm_socp_lp_solvers import SVMLp
    import pandas as pd

    url = "https://raw.githubusercontent.com/mmatthieu1290/svm-socp-lp-solvers/main/Titanic.xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    svm = SVMLp(C = 1e7,eps = 1e-4,tol_select_features = 1e-3)
    svm.fit(X,y)

    print("Coefs : ",svm.coef_)
    print("Selected features : ",svm.selected_feature_names_)


    """
    

    def __init__(self,p=0.5,C=1e4,eps=1e-5,tol=1e-4,max_iter=100,tol_select_features = 1e-5):

        

        self.p = p
        self.C = C 
        self.eps = eps 
        self.tol = tol
        self.max_iter = max_iter
        self.eps = eps
        self.tol = tol
        self.max_iter = max_iter      
        self.tol_select_features = tol_select_features

    def __sklearn_tags__(self):                       # ← 4 espacios de indentación
        tags = super().__sklearn_tags__()             # ← 8 espacios
        tags.classifier_tags.multi_class = False      # ← 8 espacios
        return tags                 

    def fit(self,X,y):


        """
        Fit the Lp-SVM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Training data.

        y : array-like of shape (n_samples,)
        Binary labels. Must be {-1, +1} or {0,+1}

        Returns
        -------
        self : object
        Fitted estimator.
        """

        self._validate_params()
        rng = check_random_state(self.random_state)

        X, y = validate_data(self, X, y, ensure_all_finite=True, y_numeric=False)

        # Validación estándar sklearn
        #check_classification_targets(y)
        #self.classes_ = np.unique(y)

        y_type = type_of_target(y, input_name='y', raise_unknown=True)
        if y_type != 'binary':
           raise ValueError(
           f"Only binary classification is supported. The type of the target "
           f"is {y_type}."
    )
        self.classes_ = np.unique(y) 
        if len(self.classes_) < 2:
            raise ValueError(
            f"Classifier can't train when only one class is present. "
            f"Got class: {self.classes_}"
        )
        

        # Mapeo interno a {-1, +1}: classes_[0] -> -1, classes_[1] -> +1
        y_internal = np.where(y == self.classes_[1], 1.0, -1.0)

        m = X.shape[0]
        n = X.shape[1]
        
        w_old = rng.randn(n)

        phi_k = np.ones(n)
        err = 2 * self.tol
        iter_ = 0
        # ========= Variables =========
        w  = cp.Variable(n)
        b  = cp.Variable()
           
        xi =  cp.Variable(m,nonneg=True)
        constraints = [] 
        for row, target,xi_i in zip(X,y_internal,xi):
            constr = target @ (w @ row.reshape((-1,1)) + b) >=  1 - xi_i
            constraints.append(constr) 


        self.n_non_zeros_coefs_per_iteration_ = []    
 
        while (err > self.tol and iter_ < self.max_iter):    
            
           weighted_abs = cp.multiply(phi_k, w) 
           obj = cp.Minimize(cp.norm2(weighted_abs)**2 + self.C * cp.sum(xi)) 
           # ========= Resolver =========
           prob = cp.Problem(obj, constraints)
           prob.solve()   
           err = npl.norm(w.value - w_old,np.inf) 
           w_old = w.value
           b_old = b.value
           xi_old = xi.value
           phi_k = np.sqrt(self.p/2)*(np.abs(w_old)**2+self.eps) ** ((self.p-2)/4)        

           self.n_non_zeros_coefs_per_iteration_.append(int((np.abs(w_old) > \
                                                            self.tol_select_features).sum()))                    

           iter_ += 1
            
        self.coef_ = w_old
        self.intercept_ = b_old
        self.xi_ = xi_old 

        self.n_iter_ = iter_ 

        self.n_non_zeros_coefs_per_iteration_ = np.array(self.n_non_zeros_coefs_per_iteration_)


        mask_selected_features = np.abs(w_old) > self.tol_select_features
        self.n_selected_features_ = int(mask_selected_features.sum())


        if hasattr(self,"feature_names_in_"):
            self.selected_feature_names_ = np.array(self.feature_names_in_)[mask_selected_features]
        
    def predict(self, X):
       """
       Predict class labels for samples in X.

       Parameters
       ----------
       X : array-like of shape (n_samples, n_features)

       Returns
       -------
       y_pred : ndarray of shape (n_samples,)
        Predicted labels in the same encoding as `classes_`.
       """
       check_is_fitted(self)
       X = validate_data(self, X, reset=False)
       scores = X @ self.coef_ + self.intercept_
       return np.where(scores >= 0, self.classes_[1], self.classes_[0])


def predict_proba(self, X):
    """
    Predict pseudo-probabilities for class labels.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    y_pred_prob : ndarray of shape (n_samples, 2)
        Column 0: pseudo-probability of `classes_[0]`.
        Column 1: pseudo-probability of `classes_[1]`.

    Notes
    -----
    These are not calibrated probabilities. They are obtained by applying a
    logistic transform to the decision function. For calibrated probabilities,
    wrap this estimator with `sklearn.calibration.CalibratedClassifierCV`.
    """
    check_is_fitted(self)
    X = validate_data(self, X, reset=False)
    scores = X @ self.coef_ + self.intercept_
    p_pos = 1.0 / (1.0 + np.exp(-scores))
    return np.column_stack([1.0 - p_pos, p_pos])