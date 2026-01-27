import numpy as np
import numpy.linalg as npl
import cvxpy as cp
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from .utils import prediction_from_w_b,prediction_probas_from_w_b

class SOCP_Lp(BaseEstimator, ClassifierMixin):

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

    The vector :math:\mu_1 (resp. :math:\mu_2) is the mean value vector of features associated with positive (resp. negative) class.
    The matrix :math:S_j\in\mathbb{R}^{n\times m_j}, with :math:j\in\{1,2\}, satisfy \sigma_j=S_jS_j^\top, where \sigma_1 (resp. \sigma_2) is the covariance matrix of features asociated with positive (resp. negative) class.   

    The constraint set of the above optimization problem is obtained from the following constraint set thanks to the the multivariate Chebyshev inequality:

    .. math::

    \inf_{\widetilde{\bf x}_j\sim ({\bm\mu}_j,\Sigma_j)} \!\!\! \text{Pr}\{(-1)^{j+1}({\bf w}^{\top }\widetilde{\bf x}_{j}+b)\ge 0\} \geq \alpha_j, \ j=1,2, 

    The notation :math:\widetilde{\bf x}_j\sim ({\bm\mu}_j,\Sigma_j)} means that the distributions :math:\widetilde{\bf x}_j have
    associated means and covariance matrices :math:({\bm\mu}_j, \Sigma_j) for :math:j = 1, 2.

    It is a robust version of SVM_Lp.

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
        :math:`(|w_j|+\varepsilon)^p`. Not a numerical tolerance.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

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

    fitted_ : bool
        True after calling fit().

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
    """
    
    def __init__(self,p=0.5,C=10**4,alpha_1=0.5,alpha_2=0.5,eps=10**(-5),tol = 1e-4,max_iter = 100):
        
        self.fitted_ = False
        self._p = None
        self.p = p
        self._C = None
        self.C = C
        self._alpha_1 = None
        self.alpha_1 = alpha_1
        self._alpha_2 = None 
        self.alpha_2 = alpha_2   
        self._eps = None
        self.eps = eps
        self._tol = None
        self.tol = tol
        self._max_iter = None
        self.max_iter = max_iter      
        
        self.kappa1 = np.sqrt(alpha_1 / (1-alpha_1))
        self.kappa2 = np.sqrt(alpha_2 / (1-alpha_2))

    @property
    def p(self):
       return self._p

    @property 
    def C(self):
       return self._C

    @property 
    def alpha_1(self):
       return self._alpha_1

    @property 
    def alpha_2(self):
       return self._alpha_2    
    
    @property
    def eps(self):
        return self._eps
    
    @property
    def tol(self):
        return self._tol

    @property
    def max_iter(self):
        return self._max_iter    
    
    

    @p.setter
    def p(self,value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("p must be a float number.")
        elif (value<=0) or (value>=1):
            raise ValueError("p must be a real number between 0 and 1")
        else:
            self._p = value

    @C.setter
    def C(self,value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("C must be a float number.")
        elif (value<=0):
            raise ValueError("C must be a positive number")
        else:
            self._C = value

    @alpha_1.setter
    def alpha_1(self,value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("alpha_1 must be a float number.")
        elif (value<=0) or (value>=1):
            raise ValueError("alpha_1 must be a real number between 0 and 1")
        else:
            self._alpha_1 = value
            self.kappa1 = np.sqrt(value / (1-value))

    @alpha_2.setter
    def alpha_2(self,value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("alpha_2 must be a float number.")
        elif (value<=0) or (value>=1):
            raise ValueError("alpha_2 must be a real number between 0 and 1")
        else:
            self._alpha_2 = value  
            self.kappa2 = np.sqrt(value / (1-value))

    @eps.setter
    def eps(self,value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("eps must be a float number or an integer number.")
        elif (value<=0):
            raise ValueError("eps must be a positive number")
        else:
            self._eps = value

    @tol.setter
    def tol(self,value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError("tol must be a float number or an integer number.")
        elif (value<=0):
            raise ValueError("tol must be a positive number")
        else:
            self._tol = value

    @max_iter.setter
    def max_iter(self,value):
        if not isinstance(value,int):
            raise TypeError("C must be a float number or an integer number.")
        elif (value<=0):
            raise ValueError("max_iter must be a positive integer")
        else:
            self._max_iter = value                    
            
        
    def fit(self,X,y):

        """
        Fit the Lp-SVM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Training data.

        y : array-like of shape (n_samples,)
        Binary labels. Recommended: {-1, +1} or {0,+1}

        tol : float, default=1e-5

        iter_max : int, default=100

        Returns
        -------
        self : object
        Fitted estimator.
        """        

        y = y.copy()
        X = X.copy()

        try:
            feature_names = X.columns.tolist()
        except AttributeError:
            _ = 0
        
        X = check_array(X,force_all_finite=True)

        _ =  check_array(X,force_all_finite=True,ensure_2d=False)
        if isinstance(y,np.ndarray) == False:
            y = np.array(y)
            
        y = y.astype(float)
        
        self.negative_value = y.min()  
        
        
        if y.ndim == 2:
            if y.shape[1] > 1:
                raise ValueError("y's number of columns must be equal to one")

        
        y = y.reshape((-1,1))
        if X.shape[0] != y.shape[0]:
            raise ValueError("The dimensions of X and y are not consistent")
            
        if (len(np.unique(y)) != 2):
            raise ValueError("The target must be a binary variable.")

        if (set(np.unique(y)) != {0,1}) & (set(np.unique(y)) != {-1,1}):
            raise ValueError("The target must contain only -1 and 1 or 0 and 1.")
            
        self.classes_ = np.unique(y)
        y[y<=0] = -1

        m = X.shape[0]
        n = X.shape[1]

        self.n_features_in_ = n
        
        A_pos = X[(y==1).reshape((-1,))]
        A_neg = X[(y<=0).reshape((-1,))]
        
        m_pos = A_pos.shape[0]
        m_neg = A_neg.shape[0]
        
        mu1 = (1 / m_pos) * A_pos.T@np.ones((m_pos,1))
        mu2 = (1 / m_neg) * A_neg.T@np.ones((m_neg,1))
        
        S1 = (1 / np.sqrt(m_pos)) * (A_pos.T - mu1 @ np.ones((1,m_pos)))
        S2 = (1 / np.sqrt(m_neg)) * (A_neg.T - mu2 @ np.ones((1,m_neg)))
        
        
        w_old = np.random.randn(n)
        b_old = np.random.randn(1)
        
        xi_old = np.random.rand(2)

        phi_k_abs = np.ones(n)
        err = 2 * self.tol
        iter_ = 0
        # ========= Variables =========
        w  = cp.Variable(n)
        b  = cp.Variable()
        
        xi = cp.Variable(2,nonneg=True)
        #   w^T μ1 + b ≥ 1 − xi_1 + κ1 ||S1^T w||
        constr1 = self.kappa1 * cp.norm(S1.T @ w, 2) <= w @ mu1 + b - 1 + xi[0]
        # −(w^T μ2 + b) ≥ 1 − xi_2 + κ2 ||S2^T w||
        constr2 = self.kappa2 * cp.norm(S2.T @ w, 2) <= -(w @ mu2 + b) - 1 + xi[1]
        constraints = [constr1, constr2]   # (xi ≥ 0 ya está en la definición de la variable) 

        self.n_non_zeros_coef_per_iteration_ = []    
            
        while (err > self.tol and iter_ < self.max_iter):    
            
           weighted_abs = cp.multiply(phi_k_abs, w) 
           obj = cp.Minimize(cp.norm1(weighted_abs) + self.C * cp.sum(xi)) 
           # ========= Resolver =========
           prob = cp.Problem(obj, constraints)
           prob.solve(solver=cp.ECOS)   
           err = npl.norm(w.value - w_old) + npl.norm(b.value - b_old) + npl.norm(xi.value - xi_old)
           w_old = w.value
           b_old = b.value
           xi_old = xi.value
           phi_k = self.p * (np.abs(w_old)+self.eps) ** (self.p-1)
           phi_k_abs = np.abs(phi_k)          
           self.n_non_zeros_coef_per_iteration_.append(int((np.abs(w_old) > 1e-5).sum()))             
           iter_ += 1
            
        self.coef_ = w_old
        self.intercept_ = b_old
        self.xi = xi_old 
        self.fitted_ = True
        self.n_iter_ = iter_
        self.n_non_zeros_coef_per_iteration_ = np.array(self.n_non_zeros_coef_per_iteration_)		

        mask_selected_features = np.abs(w_old) > 1e-5
        self.n_selected_features_ = int(mask_selected_features.sum())

        try: 
            self.feature_names_in_ = np.array(feature_names)
        except NameError:
            self.feature_names_in_ = None

        try: 
            self.selected_feature_names_ = self.feature_names_in_[mask_selected_features]
        except TypeError:
            self.selected_feature_names_ = None  
        

    
    def predict(self,X,threshold = 0.5):    
       
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

       X = X.copy() 
        
       if self.fitted_ == False:
          error_msg =  "This instance of Lp_SVM instance is not fitted yet. "
          error_msg +=  "Call 'fit' with appropriate arguments before using this estimator."
          raise NotFittedError(error_msg)

       predictions =  prediction_from_w_b(self.coef_,self.intercept_,\
                                          X,threshold,self.negative_value)    
    
       return predictions
    
    def predict_proba(self,X):

       X = X.copy() 

       if self.fitted_ == False:
          error_msg =  "This instance of Lp_SVM instance is not fitted yet. "
          error_msg +=  "Call 'fit' with appropriate arguments before using this estimator."
          raise NotFittedError(error_msg) 
       
       probas = prediction_probas_from_w_b(w=self.coef_,b=self.intercept_,X=X)
    
       return probas   
