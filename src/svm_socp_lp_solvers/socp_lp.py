import numpy as np
import numpy.linalg as npl
import cvxpy as cp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import  check_is_fitted
from numbers import Real, Integral
from sklearn.utils._param_validation import Interval
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.utils.multiclass import type_of_target


class SOCPLp(ClassifierMixin,BaseEstimator):

    _parameter_constraints = {
        "p": [Interval(Real, 0, 1, closed="neither")],
        "C": [Interval(Real, 0, None, closed="neither")],
        "alpha_1": [Interval(Real, 0, 1, closed="neither")],
        "alpha_2": [Interval(Real, 0, 1, closed="neither")],
        "tau": [Interval(Real, 0, None, closed="neither"), None],
        "eps": [Interval(Real, 0, None, closed="neither")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol_select_features": [Interval(Real, 0, None, closed="neither")],
        "random_state": ["random_state"],
    }

    r"""
    Smoothed sparse Lp-SOCP classifier.

    This estimator solves the following optimization problem:

    .. math::
        \min_{w,b,\xi}\ \sum_{j=1}^n (|w_j|+\varepsilon)^p \;+\; C\sum_{i=1}^2 \xi_i
        \quad \mathrm{s.t.}\quad
        \begin{aligned}
			&({\bf w}, b, \xi) \in \mathbb{R}^{n+2} \\
			&\text{s.t. } \ w^\top \mu_1 + b \geq 1 - \xi + \kappa(\alpha_1) \|S_1^\top w\|, \\
			&\quad -(w^\top \mu_2 + b) \geq 1 - \xi + \kappa(\alpha_2) \|S_2^\top w\|, \\
			&\quad \xi \geq 0
		\end{aligned}

    The vector :math:`\mu_1` (resp. :math:\mu_2) is the mean value vector of features associated with positive (resp. negative) class.
    The matrix :math:S_j\in\mathbb{R}^{n\times m_j}, with :math:j\in\{1,2\}, satisfy \sigma_j=S_jS_j^\top, where \sigma_1 (resp. \sigma_2) is the covariance matrix of features asociated with positive (resp. negative) class.   

    The constraint set of the above optimization problem is obtained from the following constraint set thanks to the the multivariate Chebyshev inequality:

    .. math::

    \inf_{\widetilde{\bf x}_j\sim ({\bm\mu}_j,\Sigma_j)} \!\!\! \text{Pr}\{(-1)^{j+1}({\bf w}^{\top }\widetilde{\bf x}_{j}+b)\ge 0\} \geq \alpha_j, \ j=1,2, 

    The notation :math:\widetilde{\bf x}_j\sim ({\bm\mu}_j,\Sigma_j)} means that the distributions :math:\widetilde{\bf x}_j have
    associated means and covariance matrices :math:({\bm\mu}_j, \Sigma_j) for :math:j = 1, 2.

    It is a robust version of SVMLp.

    The smoothing parameter :math:`\varepsilon>0` makes the objective locally
    Lipschitz and avoids singular behavior at :math:`w_j=0`.

    Parameters
    ----------
    p : float, default=0.5
        Exponent controlling sparsity. Must satisfy 0 < p < 1.

    C : float, default=1e4
        Slack penalty parameter. Must be > 0.

    alpha_1 : float, default=0.5
              Exponent controlling probability of good classification of positive class. Must satisfy 0 < alpha_1 < 1.

    alpha_2 : float, default=0.5   
              Exponent controlling probability of good classification of negative class. Must satisfy 0 < alpha_2 < 1.
              
    epsilon : float, default=1e-5
        Smoothing/approximation parameter :math:`\varepsilon>0` used in
        :math:`(|w_j|+\varepsilon)^p`.

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

    from svm_socp_lp_solvers import SOCPLp
    import pandas as pd
    
    url = "https://raw.githubusercontent.com/mmatthieu1290/svm-socp-lp-solvers/main/Titanic.xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    socp = SOCPLp(p=0.1,alpha_1=0.2,alpha_2=0.2)
    socp.fit(X,y)

    print("Coefs : ",socp.coef_)
    print("Selected features : ",socp.selected_feature_names_)

    """
    

    def __init__(self, p=0.5, C=1e4, alpha_1=0.5, alpha_2=0.5, tau=None,
                 eps=1e-5, tol=1e-3, max_iter=100,
                 tol_select_features=1e-5, random_state=None):
        self.p = p
        self.C = C
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.tau = tau
        self.eps = eps
        self.tol = tol
        self.max_iter = max_iter
        self.tol_select_features = tol_select_features
        self.random_state = random_state                
            
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
        Binary labels. Recommended: {-1, +1} or {0,+1}

        Returns
        -------
        self : object
        Fitted estimator.
        """        

        self._validate_params()
        rng = check_random_state(self.random_state)
        kappa1 = np.sqrt(self.alpha_1 / (1-self.alpha_1))
        kappa2 = np.sqrt(self.alpha_2 / (1-self.alpha_2))

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

        n = X.shape[1]
        
        A_pos = X[(y_internal==1).reshape((-1,))]
        A_neg = X[(y_internal<=0).reshape((-1,))]
        
        m_pos = A_pos.shape[0]
        m_neg = A_neg.shape[0]
        
        mu1 = (1 / m_pos) * A_pos.T@np.ones((m_pos,1))
        mu2 = (1 / m_neg) * A_neg.T@np.ones((m_neg,1))
        
        S1 = (1 / np.sqrt(m_pos)) * (A_pos.T - mu1 @ np.ones((1,m_pos)))
        S2 = (1 / np.sqrt(m_neg)) * (A_neg.T - mu2 @ np.ones((1,m_neg)))
        
        
        w_old = rng.randn(n)

        phi_k_abs = np.ones(n)
        err = 2 * self.tol
        iter_ = 0
        # ========= Variables =========
        w  = cp.Variable(n)
        b  = cp.Variable()
        
        xi = cp.Variable(2,nonneg=True)
        #   w^T μ1 + b ≥ 1 − xi_1 + κ1 ||S1^T w||
        constr1 = kappa1 * cp.norm(S1.T @ w, 2) <= w @ mu1 + b - 1 + xi[0]
        # −(w^T μ2 + b) ≥ 1 − xi_2 + κ2 ||S2^T w||
        constr2 = kappa2 * cp.norm(S2.T @ w, 2) <= -(w @ mu2 + b) - 1 + xi[1]
        if self.tau:
           constr3 = w @ mu1 + b <= 1 + xi[0]/self.tau
           constr4 = -self.tau *(w @ mu2 + b) <= self.tau+xi[1]
           constraints = [constr1, constr2,constr3,constr4]   # (xi ≥ 0 ya está en la definición de la variable)
        else:
           constraints = [constr1, constr2]   # (xi ≥ 0 ya está en la definición de la variable)      

        self.n_non_zeros_coef_per_iteration_ = []    
            
        while (err > self.tol and iter_ < self.max_iter):    
            
           weighted_abs = cp.multiply(phi_k_abs, w) 
           obj = cp.Minimize(cp.norm1(weighted_abs) + self.C * cp.sum(xi)) 
           # ========= Resolver =========
           prob = cp.Problem(obj, constraints)
           prob.solve(solver=cp.ECOS)   
           err = npl.norm(w.value - w_old,np.inf) 
           w_old = w.value
           b_old = b.value
           xi_old = xi.value
           phi_k = self.p * (np.abs(w_old)+self.eps) ** (self.p-1)
           phi_k_abs = np.abs(phi_k)          
           self.n_non_zeros_coef_per_iteration_.append(int((np.abs(w_old) > \
                                                            self.tol_select_features).sum()))             
           iter_ += 1
            
        self.coef_ = w_old
        self.intercept_ = b_old
        self.xi_ = xi_old 
        self.n_iter_ = iter_
        self.n_non_zeros_coef_per_iteration_ = np.array(self.n_non_zeros_coef_per_iteration_)		

        mask_selected_features = np.abs(w_old) > self.tol_select_features
        self.n_selected_features_ = int(mask_selected_features.sum())
  
        if hasattr(self,"feature_names_in_"):
            self.selected_feature_names_ = np.array(self.feature_names_in_)[mask_selected_features]
        
        return self

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