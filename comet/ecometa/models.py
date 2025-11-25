import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t
import warnings

class ThreeLevelMeta:
    """
    Three-Level Random Effects Meta-Analysis (Hierarchical Model).
    
    Models the dependency of multiple effect sizes within studies.
    Structure:
        y_ij = mu + u_i + w_ij + e_ij
    Where:
        u_i ~ N(0, tau^2)   : Between-study heterogeneity
        w_ij ~ N(0, sigma^2): Within-study heterogeneity
        e_ij ~ N(0, v_ij)   : Sampling error (known)
    """

    def __init__(self, data, effect_col, var_col, study_id_col):
        """
        Initialize the model with data.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset containing effect sizes and moderators.
        effect_col : str
            Column name for effect sizes (e.g., 'lnRR', 'hedges_g').
        var_col : str
            Column name for sampling variances.
        study_id_col : str
            Column name for the grouping variable (Study ID).
        """
        # Data Validation
        if not all(c in data.columns for c in [effect_col, var_col, study_id_col]):
            raise ValueError(f"Columns {effect_col}, {var_col}, or {study_id_col} not found in data.")

        self.data = data.copy()
        self.effect_col = effect_col
        self.var_col = var_col
        self.study_id_col = study_id_col
        
        # Pre-process data into groups for faster iteration
        # We convert to list of arrays to avoid pandas indexing overhead during optimization
        self._grouped_data = []
        grouped = self.data.groupby(study_id_col)
        
        for _, group in grouped:
            y_i = group[effect_col].values
            v_i = group[var_col].values
            self._grouped_data.append((y_i, v_i))
            
        self.n_studies = len(self._grouped_data)
        self.n_obs = len(self.data)
        self.results = None  # Stores fit results

    def _nll_reml(self, params):
        """
        Negative Log-Likelihood (REML) objective function.
        Uses Sherman-Morrison formula for efficient matrix inversion.
        """
        tau2, sigma2 = params
        
        # Bounds protection (optimizers sometimes drift slightly negative)
        if tau2 < 0: tau2 = 1e-10
        if sigma2 < 0: sigma2 = 1e-10

        sum_log_det = 0.0
        sum_S = 0.0       # 1' * V^-1 * 1
        sum_Sy = 0.0      # 1' * V^-1 * y
        sum_ySy = 0.0     # y' * V^-1 * y

        for y_i, v_i in self._grouped_data:
            # V_i = D + sigma2*I + tau2*J
            # A = D + sigma2*I (Diagonal matrix)
            A_diag = v_i + sigma2
            inv_A_diag = 1.0 / A_diag
            
            # Sherman-Morrison components
            sum_inv_A = np.sum(inv_A_diag)
            denom = 1 + tau2 * sum_inv_A
            
            # Log Determinant
            # det(A + uv^T) = det(A) * (1 + v^T A^-1 u)
            log_det_A = np.sum(np.log(A_diag))
            log_det_Vi = log_det_A + np.log(denom)
            sum_log_det += log_det_Vi

            # Weights calculation using Woodbury identity
            # w_y = V^-1 * y
            inv_A_y = inv_A_diag * y_i
            sum_inv_A_y = np.sum(inv_A_y)
            
            w_y = inv_A_y - (tau2 * inv_A_diag * sum_inv_A_y) / denom
            
            # w_1 = V^-1 * 1
            w_1 = inv_A_diag - (tau2 * inv_A_diag * sum_inv_A) / denom

            # Accumulate sums
            sum_S += np.sum(w_1)
            sum_Sy += np.sum(w_y)
            sum_ySy += np.dot(y_i, w_y)

        # Check for singular matrix
        if sum_S <= 1e-10:
            return 1e10

        # Profiled Mean
        mu_hat = sum_Sy / sum_S
        
        # Residual Sum of Squares (Generalized)
        # (y - Xb)' V^-1 (y - Xb)
        rss = sum_ySy - 2.0 * mu_hat * sum_Sy + mu_hat**2 * sum_S

        # REML Log-Likelihood
        # Constant term excluded as it doesn't affect minimization
        nll = 0.5 * (sum_log_det + np.log(sum_S) + rss)
        
        return nll

    def fit(self, verbose=False):
        """
        Fit the model using a two-pass optimization strategy (Global -> Local).
        
        Returns
        -------
        self : Returns the instance itself.
        """
        if self.n_studies < 2:
            raise ValueError("Insufficient studies (k<2) for 3-level meta-analysis.")

        # 1. Strategy: Multiple start points to avoid local minima
        # [tau2, sigma2] pairs
        start_points = [
            [0.01, 0.01], 
            [0.1, 0.1], 
            [0.5, 0.01], 
            [0.01, 0.5],
            [np.var(self.data[self.effect_col])/2, 0.01] # Data-driven guess
        ]
        
        bounds = [(1e-8, None), (1e-8, None)]
        best_res = None
        best_fun = np.inf

        # Pass 1: L-BFGS-B (Global Search)
        for start in start_points:
            try:
                res = minimize(
                    self._nll_reml, 
                    x0=start, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'ftol': 1e-10}
                )
                if res.success and res.fun < best_fun:
                    best_fun = res.fun
                    best_res = res
            except Exception:
                continue

        if best_res is None:
            raise RuntimeError("Optimization failed to converge.")

        # Pass 2: Nelder-Mead (Polishing the result)
        # Useful for flat likelihood surfaces common in variance estimation
        final_res = minimize(
            self._nll_reml, 
            x0=best_res.x, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={'xatol': 1e-10, 'fatol': 1e-10}
        )

        self._compute_statistics(final_res.x, final_res.fun)
        return self

    def _compute_statistics(self, variance_params, nll_val):
        """
        Compute final statistics (Mu, SE, CI, I2) based on optimal variances.
        """
        tau2, sigma2 = variance_params
        
        # Re-run the core logic one last time to get sums
        sum_S = 0.0
        sum_Sy = 0.0
        
        for y_i, v_i in self._grouped_data:
            A_diag = v_i + sigma2
            inv_A_diag = 1.0 / A_diag
            sum_inv_A = np.sum(inv_A_diag)
            denom = 1 + tau2 * sum_inv_A
            
            # Weights for Mu
            inv_A_y = inv_A_diag * y_i
            sum_inv_A_y = np.sum(inv_A_y)
            
            w_y = inv_A_y - (tau2 * inv_A_diag * sum_inv_A_y) / denom
            w_1 = inv_A_diag - (tau2 * inv_A_diag * sum_inv_A) / denom
            
            sum_S += np.sum(w_1)
            sum_Sy += np.sum(w_y)

        mu_hat = sum_Sy / sum_S
        var_mu = 1.0 / sum_S
        se_mu = np.sqrt(var_mu)
        
        # Statistics
        z_score = mu_hat / se_mu
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        # Heterogeneity (I2)
        # For 3-level, we define I2_total, I2_between, I2_within
        # Standard "typical" sampling variance approximation
        wi_typical = 1 / self.data[self.var_col].values
        k = len(wi_typical)
        # Typical sampling variance (Higgins & Thompson 2002)
        v_typical = (k - 1) / (np.sum(wi_typical) - np.sum(wi_typical**2)/np.sum(wi_typical))
        
        total_var = tau2 + sigma2 + v_typical
        i2_total = ((tau2 + sigma2) / total_var) * 100
        i2_between = (tau2 / total_var) * 100
        i2_within = (sigma2 / total_var) * 100

        # AIC (approximate)
        # Params = 3 (mu, tau2, sigma2)
        log_lik = -nll_val # Since we minimized negative log lik
        aic = 2 * 3 - 2 * log_lik

        self.results = {
            'pooled_effect': mu_hat,
            'se': se_mu,
            'z_score': z_score,
            'p_value': p_value,
            'ci_lower': mu_hat - 1.96 * se_mu,
            'ci_upper': mu_hat + 1.96 * se_mu,
            'tau2': tau2,
            'sigma2': sigma2,
            'i2_total': i2_total,
            'i2_between': i2_between,
            'i2_within': i2_within,
            'aic': aic,
            'log_likelihood': log_lik
        }

    def summary(self):
        """
        Returns a Pandas Series containing the results.
        """
        if self.results is None:
            raise ValueError("Model not fitted. Run .fit() first.")
        
        return pd.Series(self.results, name="3-Level Meta-Analysis Results")