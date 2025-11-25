"""
Meta-Regression and Spline Models for Three-Level Meta-Analysis.

This module extends the basic three-level meta-analysis model to include
moderator variables (covariates) using both linear and non-linear (spline) approaches.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
from scipy.optimize import minimize
from scipy.stats import t, chi2, norm
from .models import ThreeLevelMeta

# Try importing patsy for splines
try:
    import patsy
    PATSY_AVAILABLE = True
except ImportError:
    PATSY_AVAILABLE = False
    warnings.warn("patsy not installed. Spline regression will not be available. "
                  "Install with: pip install patsy")


class MetaRegression(ThreeLevelMeta):
    """
    Three-Level Mixed-Effects Meta-Regression.

    Extends the standard random-effects model to include moderator variables.
    Model: y = Xβ + u + w + e

    where:
        y: effect sizes
        X: design matrix (includes intercept + moderators)
        β: regression coefficients
        u ~ N(0, τ²): between-study random effects
        w ~ N(0, σ²): within-study random effects
        e ~ N(0, v): sampling error (known)

    This implementation uses robust variance estimation with Sherman-Morrison
    matrix inversion for numerical stability.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing effect sizes and moderators
    effect_col : str
        Column name for effect sizes
    var_col : str
        Column name for sampling variances
    study_id_col : str
        Column name for the grouping variable (Study ID)
    moderators : str or list of str
        Column name(s) for moderator variable(s)

    Examples
    --------
    >>> import pandas as pd
    >>> from ecometa import MetaRegression
    >>> data = pd.DataFrame({
    ...     'effect': [0.5, 0.8, 0.3, 0.6],
    ...     'variance': [0.1, 0.12, 0.09, 0.11],
    ...     'study_id': ['A', 'A', 'B', 'B'],
    ...     'year': [2010, 2010, 2015, 2015]
    ... })
    >>> model = MetaRegression(data, 'effect', 'variance', 'study_id', 'year')
    >>> model.fit()
    >>> print(model.summary())
    """

    def __init__(self, data, effect_col, var_col, study_id_col, moderators):
        """Initialize meta-regression model."""
        super().__init__(data, effect_col, var_col, study_id_col)

        # Handle single moderator or multiple moderators
        if isinstance(moderators, str):
            self.moderators = [moderators]
        else:
            self.moderators = list(moderators)

        # Validate moderator columns exist
        for mod in self.moderators:
            if mod not in self.data.columns:
                raise ValueError(f"Moderator column '{mod}' not found in data.")

        # Prepare Design Matrix X (Intercept + Moderators)
        self.X = self._create_design_matrix()
        self.p_params = self.X.shape[1]  # Number of parameters (including intercept)

        # Group X by study for the optimizer (matches parent class grouping)
        self._grouped_X = []
        grouped = self.data.groupby(study_id_col)
        for study_id, group in grouped:
            X_i = self._create_design_matrix(group)
            self._grouped_X.append(X_i)

    def _create_design_matrix(self, subset_data=None):
        """
        Create design matrix with intercept and moderator variables.

        Parameters
        ----------
        subset_data : pd.DataFrame, optional
            If provided, creates design matrix for this subset

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (n_obs, p_params)
        """
        if subset_data is None:
            subset_data = self.data

        # Extract moderator values
        X_mods = subset_data[self.moderators].values

        # Add intercept
        X = sm.add_constant(X_mods, prepend=True)

        return X

    def _nll_regression(self, params):
        """
        Negative Log-Likelihood for Meta-Regression (REML).

        Uses Sherman-Morrison inversion for robust computation.

        Parameters
        ----------
        params : array-like
            [tau2, sigma2] variance parameters

        Returns
        -------
        nll : float
            Negative REML log-likelihood
        """
        tau2, sigma2 = params
        # Bounds protection
        if tau2 < 0:
            tau2 = 1e-10
        if sigma2 < 0:
            sigma2 = 1e-10

        sum_log_det_Vi = 0.0
        sum_XWX = np.zeros((self.p_params, self.p_params))
        sum_XWy = np.zeros(self.p_params)
        sum_yWy = 0.0

        # Iterate over studies
        for i, (y_i, v_i) in enumerate(self._grouped_data):
            X_i = self._grouped_X[i]

            # V_i = D + sigma2*I + tau2*J
            # A = D + sigma2*I (Diagonal matrix)
            A_diag = v_i + sigma2
            inv_A_diag = 1.0 / A_diag

            # Sherman-Morrison components
            sum_inv_A = np.sum(inv_A_diag)
            denom = 1 + tau2 * sum_inv_A

            # Log Determinant of V_i
            # det(A + uv^T) = det(A) * (1 + v^T A^-1 u)
            log_det_A = np.sum(np.log(A_diag))
            sum_log_det_Vi += log_det_A + np.log(denom)

            # Matrix operations for X'V^-1X without full inversion
            # V^-1 = A^-1 - (tau2 / denom) * (A^-1 1)(1' A^-1)

            # Precompute A^-1 * X and A^-1 * y (element-wise since A is diagonal)
            inv_A_X = inv_A_diag[:, None] * X_i
            inv_A_y = inv_A_diag * y_i

            # Column sums (equivalent to 1' A^-1 X)
            sum_inv_A_X = np.sum(inv_A_X, axis=0)
            sum_inv_A_y = np.sum(inv_A_y)

            # Update global sums for X' W X
            # Term 1: X' A^-1 X
            xt_invA_x = X_i.T @ inv_A_X
            # Term 2: Sherman-Morrison correction
            correction_term = (tau2 / denom) * np.outer(sum_inv_A_X, sum_inv_A_X)
            sum_XWX += (xt_invA_x - correction_term)

            # Update global X' W y
            xt_invA_y = X_i.T @ inv_A_y
            correction_y = (tau2 / denom) * sum_inv_A_X * sum_inv_A_y
            sum_XWy += (xt_invA_y - correction_y)

            # Update global y' W y
            yt_invA_y = np.dot(y_i, inv_A_y)
            correction_yy = (tau2 / denom) * (sum_inv_A_y**2)
            sum_yWy += (yt_invA_y - correction_yy)

        # Solve GLS equations for betas: (X'V^-1X) β = X'V^-1y
        try:
            betas = np.linalg.solve(sum_XWX, sum_XWy)
        except np.linalg.LinAlgError:
            return 1e10

        # Residual Sum of Squares: y'V^-1y - β'X'V^-1y
        rss = sum_yWy - np.dot(betas, sum_XWy)

        # REML Log-Likelihood
        # L = -0.5 * [log|V| + log|X'V^-1X| + RSS]
        sign, log_det_XWX = np.linalg.slogdet(sum_XWX)
        if sign <= 0:
            return 1e10

        nll = 0.5 * (sum_log_det_Vi + log_det_XWX + rss)
        return nll

    def fit(self, verbose=False):
        """
        Fit the meta-regression model using REML estimation.

        Uses a multi-start optimization strategy to avoid local minima.

        Parameters
        ----------
        verbose : bool, default=False
            If True, prints optimization progress

        Returns
        -------
        self : Returns the instance itself
        """
        if self.n_studies < 2:
            raise ValueError("Insufficient studies (k<2) for meta-regression.")

        # Multiple start points for robust optimization
        start_points = [
            [0.01, 0.01],
            [0.1, 0.1],
            [0.5, 0.1],
            [0.1, 0.5],
            [1.0, 0.1],
            [0.01, 1.0]
        ]

        bounds = [(1e-8, None), (1e-8, None)]
        best_res = None
        best_fun = np.inf

        # Pass 1: L-BFGS-B (Global search)
        for start in start_points:
            try:
                res = minimize(
                    self._nll_regression,
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
            raise RuntimeError("Meta-regression optimization failed to converge.")

        # Pass 2: Nelder-Mead (Polish the result)
        final_res = minimize(
            self._nll_regression,
            x0=best_res.x,
            method='Nelder-Mead',
            bounds=bounds,
            options={'xatol': 1e-10, 'fatol': 1e-10}
        )

        # Compute final statistics at optimal variance parameters
        tau2, sigma2 = final_res.x
        params, cov_params = self._compute_gls_matrices(tau2, sigma2)

        # Store results
        self._store_results(params, cov_params, tau2, sigma2, final_res.fun)

        return self

    def _compute_gls_matrices(self, tau2, sigma2):
        """
        Compute regression coefficients and covariance matrix at given variance values.

        Parameters
        ----------
        tau2 : float
            Between-study variance
        sigma2 : float
            Within-study variance

        Returns
        -------
        betas : np.ndarray
            Regression coefficients
        var_betas : np.ndarray
            Covariance matrix of coefficients
        """
        sum_XWX = np.zeros((self.p_params, self.p_params))
        sum_XWy = np.zeros(self.p_params)

        for i, (y_i, v_i) in enumerate(self._grouped_data):
            X_i = self._grouped_X[i]

            # V_i components
            A_diag = v_i + sigma2
            inv_A_diag = 1.0 / A_diag
            sum_inv_A = np.sum(inv_A_diag)
            denom = 1 + tau2 * sum_inv_A

            # Compute weighted matrices
            inv_A_X = inv_A_diag[:, None] * X_i
            inv_A_y = inv_A_diag * y_i
            sum_inv_A_X = np.sum(inv_A_X, axis=0)
            sum_inv_A_y = np.sum(inv_A_y)

            # Accumulate X'WX and X'Wy
            xt_invA_x = X_i.T @ inv_A_X
            correction_term = (tau2 / denom) * np.outer(sum_inv_A_X, sum_inv_A_X)
            sum_XWX += (xt_invA_x - correction_term)

            xt_invA_y = X_i.T @ inv_A_y
            correction_y = (tau2 / denom) * sum_inv_A_X * sum_inv_A_y
            sum_XWy += (xt_invA_y - correction_y)

        # Solve for betas and compute covariance
        betas = np.linalg.solve(sum_XWX, sum_XWy)
        var_betas = np.linalg.inv(sum_XWX)

        return betas, var_betas

    def _store_results(self, betas, cov_betas, tau2, sigma2, nll_val):
        """Store regression results in a structured format."""
        # Standard errors
        se_betas = np.sqrt(np.diag(cov_betas))

        # Z-statistics and p-values
        z_stats = betas / se_betas
        p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))

        # Confidence intervals
        ci_lower = betas - 1.96 * se_betas
        ci_upper = betas + 1.96 * se_betas

        # Model fit statistics
        log_lik = -nll_val
        n_params = self.p_params + 2  # betas + tau2 + sigma2
        aic = 2 * n_params - 2 * log_lik
        bic = n_params * np.log(self.n_obs) - 2 * log_lik

        # Create coefficient table
        coef_names = ['intercept'] + self.moderators
        coef_table = pd.DataFrame({
            'coefficient': betas,
            'se': se_betas,
            'z_value': z_stats,
            'p_value': p_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }, index=coef_names)

        self.results = {
            'coefficients': coef_table,
            'tau2': tau2,
            'sigma2': sigma2,
            'tau': np.sqrt(tau2),
            'sigma': np.sqrt(sigma2),
            'log_likelihood': log_lik,
            'aic': aic,
            'bic': bic,
            'n_studies': self.n_studies,
            'n_obs': self.n_obs,
            'cov_matrix': cov_betas
        }

    def summary(self):
        """
        Return a formatted summary of the regression results.

        Returns
        -------
        summary_text : str
            Formatted summary of the model
        """
        if self.results is None:
            raise ValueError("Model not fitted. Run .fit() first.")

        coef_table = self.results['coefficients']

        summary = []
        summary.append("=" * 70)
        summary.append("Three-Level Meta-Regression Results")
        summary.append("=" * 70)
        summary.append(f"Number of studies:     {self.results['n_studies']}")
        summary.append(f"Number of observations: {self.results['n_obs']}")
        summary.append(f"Log-Likelihood:        {self.results['log_likelihood']:.4f}")
        summary.append(f"AIC:                   {self.results['aic']:.4f}")
        summary.append(f"BIC:                   {self.results['bic']:.4f}")
        summary.append("")
        summary.append("Variance Components:")
        summary.append(f"  τ² (between-study):  {self.results['tau2']:.6f}")
        summary.append(f"  σ² (within-study):   {self.results['sigma2']:.6f}")
        summary.append("")
        summary.append("Coefficients:")
        summary.append("-" * 70)
        summary.append(f"{'Variable':<15} {'Coef':>10} {'SE':>10} {'z':>8} {'P>|z|':>10} {'[95% CI]':>20}")
        summary.append("-" * 70)

        for idx, row in coef_table.iterrows():
            ci_str = f"[{row['ci_lower']:6.3f}, {row['ci_upper']:6.3f}]"
            p_str = f"{row['p_value']:.4f}" if row['p_value'] >= 0.001 else "< 0.001"
            summary.append(
                f"{idx:<15} {row['coefficient']:10.4f} {row['se']:10.4f} "
                f"{row['z_value']:8.3f} {p_str:>10} {ci_str:>20}"
            )

        summary.append("=" * 70)

        return "\n".join(summary)


class SplineMetaRegression:
    """
    Non-linear Meta-Regression using Natural Cubic Splines.

    Fits a smooth non-linear relationship between a continuous moderator
    and effect sizes. Uses the 'plug-in τ²' method for stability: estimates
    τ² from a linear model, then fixes it for the spline model.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing effect sizes and moderators
    effect_col : str
        Column name for effect sizes
    var_col : str
        Column name for sampling variances
    study_id_col : str
        Column name for study ID (for aggregation)
    moderator_col : str
        Column name for the continuous moderator variable
    df_spline : int, default=3
        Degrees of freedom for the natural cubic spline
        (controls smoothness; higher = more flexible)

    Examples
    --------
    >>> from ecometa import SplineMetaRegression
    >>> model = SplineMetaRegression(data, 'effect', 'variance', 'study_id',
    ...                               'temperature', df_spline=4)
    >>> model.fit()
    >>> model.plot()
    """

    def __init__(self, data, effect_col, var_col, study_id_col, moderator_col, df_spline=3):
        """Initialize spline meta-regression model."""
        if not PATSY_AVAILABLE:
            raise ImportError("patsy is required for spline regression. "
                              "Install with: pip install patsy")

        # Validate inputs
        required_cols = [effect_col, var_col, study_id_col, moderator_col]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        # Remove missing values
        self.data = data.dropna(subset=required_cols).copy()

        if len(self.data) < df_spline + 2:
            raise ValueError(f"Insufficient data points for df_spline={df_spline}. "
                             f"Need at least {df_spline + 2} observations.")

        self.effect_col = effect_col
        self.var_col = var_col
        self.study_id_col = study_id_col
        self.moderator_col = moderator_col
        self.df_spline = df_spline
        self.results = None

    def fit(self, fixed_tau2=None):
        """
        Fit the spline meta-regression model.

        Parameters
        ----------
        fixed_tau2 : float, optional
            If provided, uses this value for τ².
            If None, estimates τ² from a linear random-effects model.

        Returns
        -------
        self : Returns the instance itself
        """
        # 1. Aggregate data to study level (splines unstable with full 3-level)
        agg_data = self._aggregate_to_study_level()

        # 2. Estimate τ² from linear model if not provided
        if fixed_tau2 is None:
            fixed_tau2 = self._estimate_tau2_linear(agg_data)

        # 3. Extract aggregated values
        y = agg_data['mean_effect'].values
        v = agg_data['mean_variance'].values
        x = agg_data['mean_moderator'].values

        # 4. Standardize moderator for numerical stability
        self.x_mean = x.mean()
        self.x_std = x.std()
        x_standardized = (x - self.x_mean) / self.x_std

        # 5. Create spline basis using patsy
        # cr() = natural cubic spline (restricted cubic spline)
        formula = f"cr(x, df={self.df_spline}) - 1"  # -1 removes intercept from spline basis
        spline_basis = patsy.dmatrix(formula, {"x": x_standardized}, return_type='dataframe')

        # 6. Add intercept to design matrix
        X = sm.add_constant(spline_basis, prepend=True)

        # 7. Fit WLS with fixed τ² (prevents overfitting)
        weights = 1.0 / (v + fixed_tau2)
        wls_model = sm.WLS(y, X, weights=weights).fit()

        # 8. Store results
        self.results = {
            'wls_model': wls_model,
            'tau2_plugin': fixed_tau2,
            'aic': wls_model.aic,
            'bic': wls_model.bic,
            'params': wls_model.params,
            'se_params': wls_model.bse,
            'pvalues': wls_model.pvalues,
            'formula': formula,
            'x_mean': self.x_mean,
            'x_std': self.x_std,
            'agg_data': agg_data
        }

        return self

    def _aggregate_to_study_level(self):
        """
        Aggregate multiple effect sizes within studies to study-level means.

        Uses inverse-variance weighting for aggregation.

        Returns
        -------
        agg_df : pd.DataFrame
            Study-level aggregated data
        """
        grouped = self.data.groupby(self.study_id_col)
        agg_list = []

        for study_id, group in grouped:
            # Inverse-variance weights
            weights = 1 / group[self.var_col]
            sum_weights = weights.sum()

            # Weighted mean effect
            mean_effect = (weights * group[self.effect_col]).sum() / sum_weights

            # Aggregated variance (inverse of sum of weights)
            mean_variance = 1 / sum_weights

            # Mean moderator value
            mean_moderator = group[self.moderator_col].mean()

            agg_list.append({
                'study_id': study_id,
                'mean_effect': mean_effect,
                'mean_variance': mean_variance,
                'mean_moderator': mean_moderator,
                'n_effects': len(group)
            })

        return pd.DataFrame(agg_list)

    def _estimate_tau2_linear(self, agg_data):
        """
        Estimate τ² from a simple linear random-effects model.

        Uses DerSimonian-Laird estimator on study-level aggregated data.

        Parameters
        ----------
        agg_data : pd.DataFrame
            Study-level aggregated data

        Returns
        -------
        tau2 : float
            Estimated between-study variance
        """
        from .estimators import calculate_tau_squared_DL

        tau2 = calculate_tau_squared_DL(
            agg_data,
            'mean_effect',
            'mean_variance'
        )

        return tau2

    def predict(self, x_new, ci=True, alpha=0.05):
        """
        Predict effect sizes for new moderator values.

        Parameters
        ----------
        x_new : array-like
            New values of the moderator for prediction
        ci : bool, default=True
            If True, returns confidence intervals
        alpha : float, default=0.05
            Significance level for confidence intervals

        Returns
        -------
        predictions : pd.DataFrame
            Predicted effect sizes (and CIs if requested)
        """
        if self.results is None:
            raise ValueError("Model not fitted. Run .fit() first.")

        x_new = np.asarray(x_new)

        # Standardize new x values
        x_new_std = (x_new - self.x_mean) / self.x_std

        # Create spline basis for new values
        spline_basis = patsy.dmatrix(
            self.results['formula'],
            {"x": x_new_std},
            return_type='dataframe'
        )
        X_new = sm.add_constant(spline_basis, prepend=True)

        # Predict
        predictions = self.results['wls_model'].predict(X_new)

        if ci:
            # Standard errors of predictions
            se_pred = np.sqrt((X_new @ self.results['wls_model'].cov_params() * X_new).sum(axis=1))

            # Confidence intervals
            z_crit = norm.ppf(1 - alpha / 2)
            ci_lower = predictions - z_crit * se_pred
            ci_upper = predictions + z_crit * se_pred

            return pd.DataFrame({
                'moderator': x_new,
                'predicted_effect': predictions,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })
        else:
            return pd.DataFrame({
                'moderator': x_new,
                'predicted_effect': predictions
            })

    def summary(self):
        """
        Return a formatted summary of the spline regression results.

        Returns
        -------
        summary_text : str
            Formatted summary
        """
        if self.results is None:
            raise ValueError("Model not fitted. Run .fit() first.")

        summary = []
        summary.append("=" * 70)
        summary.append("Three-Level Spline Meta-Regression Results")
        summary.append("=" * 70)
        summary.append(f"Moderator:             {self.moderator_col}")
        summary.append(f"Spline degrees of freedom: {self.df_spline}")
        summary.append(f"Number of studies:     {len(self.results['agg_data'])}")
        summary.append(f"Plug-in τ²:            {self.results['tau2_plugin']:.6f}")
        summary.append(f"AIC:                   {self.results['aic']:.4f}")
        summary.append(f"BIC:                   {self.results['bic']:.4f}")
        summary.append("")
        summary.append("Note: This is a smooth non-linear model.")
        summary.append("Use .predict() to get fitted values at specific moderator levels.")
        summary.append("=" * 70)

        return "\n".join(summary)
