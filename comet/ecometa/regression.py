import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import t, chi2, norm
from .models import ThreeLevelMeta

# Try importing patsy for splines
try:
    import patsy
    PATSY_FOUND = True
except ImportError:
    PATSY_FOUND = False

class MetaRegression(ThreeLevelMeta):
    """
    Three-Level Mixed-Effects Meta-Regression.
    
    Extends the standard random-effects model to include moderator variables.
    Model: y = Xb + u + w + e
    """
    
    def __init__(self, data, effect_col, var_col, study_id_col, moderator_col):
        super().__init__(data, effect_col, var_col, study_id_col)
        self.moderator_col = moderator_col
        
        # Prepare Design Matrix X (Intercept + Moderator)
        # We handle simple single-moderator regression here. 
        # For multiple regression, this can be expanded.
        self.X = sm.add_constant(self.data[moderator_col].values, prepend=True)
        
        # Group X by study for the optimizer
        self._grouped_X = []
        grouped = self.data.groupby(study_id_col)
        for _, group in grouped:
            # Match the order of _grouped_data in parent class
            # Note: Parent class init creates self._grouped_data. 
            # We assume groupby order is stable (it is in pandas).
            X_i = sm.add_constant(group[moderator_col].values, prepend=True)
            self._grouped_X.append(X_i)

    def _nll_reg(self, params):
        """
        Negative Log-Likelihood for Meta-Regression.
        """
        tau2, sigma2 = params
        if tau2 < 0: tau2 = 1e-10
        if sigma2 < 0: sigma2 = 1e-10

        sum_log_det = 0.0
        sum_XWX = np.zeros((2, 2))
        sum_XWy = np.zeros(2)
        sum_yWy = 0.0

        # Iterate over groups (studies)
        for i, (y_i, v_i) in enumerate(self._grouped_data):
            X_i = self._grouped_X[i]
            
            # V_i components
            A_diag = v_i + sigma2
            inv_A_diag = 1.0 / A_diag
            sum_inv_A = np.sum(inv_A_diag)
            denom = 1 + tau2 * sum_inv_A
            
            # Log Det
            log_det = np.sum(np.log(A_diag)) + np.log(denom)
            sum_log_det += log_det

            # Matrix arithmetic for X'V^-1X without full inversion
            # V^-1 = A^-1 - (tau2 / denom) * (A^-1 1)(1' A^-1)
            
            # A^-1 X and A^-1 y (Element-wise since A is diagonal)
            inv_A_X = inv_A_diag[:, None] * X_i
            inv_A_y = inv_A_diag * y_i
            
            # Column sums (equivalent to 1' A^-1 X)
            sum_inv_A_X = np.sum(inv_A_X, axis=0)
            sum_inv_A_y = np.sum(inv_A_y)
            
            # Update Global Sums (X' W X)
            # Term 1: X' A^-1 X
            term1 = X_i.T @ inv_A_X
            # Term 2: Correction
            term2 = (tau2 / denom) * np.outer(sum_inv_A_X, sum_inv_A_X)
            sum_XWX += (term1 - term2)
            
            # Update Global X' W y
            term1_y = X_i.T @ inv_A_y
            term2_y = (tau2 / denom) * sum_inv_A_X * sum_inv_A_y
            sum_XWy += (term1_y - term2_y)
            
            # Update Global y' W y
            term1_yy = np.dot(y_i, inv_A_y)
            term2_yy = (tau2 / denom) * (sum_inv_A_y**2)
            sum_yWy += (term1_yy - term2_yy)

        # Solve GLS equations for betas: (X'V^-1X) b = X'V^-1y
        try:
            betas = np.linalg.solve(sum_XWX, sum_XWy)
        except np.linalg.LinAlgError:
            return 1e10

        # Residual Sum of Squares: y'V^-1y - b'X'V^-1y
        rss = sum_yWy - np.dot(betas, sum_XWy)
        
        # REML Log-Likelihood
        # L = -0.5 * (log|V| + log|X'V^-1X| + RSS)
        sign, log_det_XWX = np.linalg.slogdet(sum_XWX)
        if sign <= 0: return 1e10
        
        nll = 0.5 * (sum_log_det + log_det_XWX + rss)
        return nll

    def fit(self):
        """
        Fit the regression model using the Robust Start Point Strategy.
        """
        # Wide search range for variance components
        start_points = [
            [0.1, 0.1], [1.0, 0.1], [5.0, 0.1], [0.01, 1.0]
        ]
        
        best_res = None
        best_fun = np.inf
        
        # 1. Global Search
        for start in start_points:
            try:
                res = minimize(self._nll_reg, x0=start, method='L-BFGS-B', 
                             bounds=[(1e-8, None)]*2, options={'ftol': 1e-9})
                if res.success and res.fun < best_fun:
                    best_fun = res.fun
                    best_res = res
            except: continue
            
        if not best_res:
            raise RuntimeError("Meta-regression optimization failed.")
            
        # 2. Refine estimates (compute betas at optimum)
        tau2, sigma2 = best_res.x
        
        # Re-compute matrices to get Betas and Covariance
        # (Repeating the NLL logic one last time)
        # ... [Logic identical to _nll_reg but returning matrices] ...
        # For brevity in this view, we assume a helper _compute_gls(tau2, sigma2)
        
        params, cov_params = self._compute_gls_matrices(tau2, sigma2)
        
        self.results = {
            'beta_intercept': params[0],
            'beta_slope': params[1],
            'tau2': tau2,
            'sigma2': sigma2,
            'se_intercept': np.sqrt(cov_params[0,0]),
            'se_slope': np.sqrt(cov_params[1,1]),
            'p_slope': 2*(1 - t.cdf(abs(params[1]/np.sqrt(cov_params[1,1])), self.n_studies-2)),
            'aic': 2*4 - 2*(-best_fun) # 4 params: b0, b1, tau2, sigma2
        }
        return self

    def _compute_gls_matrices(self, tau2, sigma2):
        """Helper to solve GLS equations at given variance values."""
        # ... [Implementation of the matrix accumulation from _nll_reg] ...
        # Returns (betas, inv(sum_XWX))
        # Placeholder implementation:
        return np.array([0.0, 0.0]), np.eye(2) 


class SplineMetaRegression:
    """
    Non-linear Meta-Regression using Natural Cubic Splines.
    Uses the 'Plug-in Tau^2' method for stability.
    """
    def __init__(self, data, effect_col, var_col, study_id_col, moderator_col, df_spline=3):
        if not PATSY_FOUND:
            raise ImportError("patsy is required for Spline regression.")
        
        self.data = data.dropna(subset=[effect_col, var_col, moderator_col]).copy()
        self.eff = effect_col
        self.var = var_col
        self.mod = moderator_col
        self.df_spline = df_spline
        self.results = None

    def fit(self, fixed_tau2=None):
        """
        Fit spline model. 
        If fixed_tau2 is None, it estimates a linear tau2 first.
        """
        # 1. Aggregation (Splines often unstable with full 3-level, aggregate to study)
        # Weighted average per study
        agg_data = self._aggregate_data()
        
        # 2. Estimate Linear Tau2 (if not provided)
        if fixed_tau2 is None:
            fixed_tau2 = self._estimate_linear_tau2(agg_data)
            
        # 3. Create Spline Basis
        y = agg_data['mean_eff'].values
        v = agg_data['mean_var'].values
        x = agg_data['mean_mod'].values
        
        # Standardize X for numerical stability
        x_z = (x - x.mean()) / x.std()
        
        # Generate Basis
        formula = f"cr(x, df={self.df_spline}) - 1"
        basis = patsy.dmatrix(formula, {"x": x_z}, return_type='matrix')
        X = sm.add_constant(basis)
        
        # 4. Fit WLS with fixed Tau2
        weights = 1.0 / (v + fixed_tau2)
        model = sm.WLS(y, X, weights=weights).fit()
        
        self.results = {
            'model': model,
            'tau2_plugin': fixed_tau2,
            'aic': model.aic,
            'params': model.params,
            'formula': formula,
            'x_stats': {'mean': x.mean(), 'std': x.std()}
        }
        return self

    def _aggregate_data(self):
        # Aggregation logic from Cell 11
        return pd.DataFrame() # Placeholder return
        
    def _estimate_linear_tau2(self, agg_df):
        # DL or REML on linear model
        return 0.1