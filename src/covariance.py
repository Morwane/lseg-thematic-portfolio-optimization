"""Factor-model covariance estimation module.

Implements the Barra-style factor model:

    r_i = sum_k(B_ik * f_k) + epsilon_i

    Sigma = B @ F_cov @ B.T + D

where:
    B       : (n x K) matrix of factor loadings (estimated via OLS)
    F_cov   : (K x K) covariance matrix of factor returns
    D       : (n x n) diagonal matrix of idiosyncratic variances
    Sigma   : (n x n) full covariance matrix

Factors are built as equal-weight sector portfolios from the universe itself —
no external data required. This makes the implementation self-contained and
reproducible.

Reference:
    Barra Risk Models (1974); Ledoit & Wolf (2003) for shrinkage alternatives.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


TRADING_DAYS = 252

# Sector membership — must stay in sync with factor_analysis.py SECTOR_MAPPING
SECTOR_MEMBERS: Dict[str, List[str]] = {
    "Semiconductors": ["NVDA.O", "AMD.O", "INTC.O", "QCOM.O", "ARM.O", "TSM.N", "ASML.O"],
    "Cloud": ["MSFT.O", "GOOGL.O", "AMZN.O", "ORCL.N", "CRM.N", "IBM.N", "ADBE.O"],
    "Software": ["META.O", "AAPL.O", "NOW.N", "SNOW.N", "PLTR.N", "MSTR.O"],
}

FACTOR_NAMES = ["Semiconductors", "Cloud", "Software"]


def build_factor_returns(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Construct observable factor return series from the asset universe.

    Uses 3 sector factors (Semiconductors, Cloud, Software) plus a Market factor.
    The Market factor is included as an intercept in OLS — NOT as an explicit
    factor series — to avoid multicollinearity (sector factors are ~99% correlated
    with Market when computed as EW averages of the same universe).

    Each sector factor is the equal-weight average return of valid tickers
    in that sector. If fewer than 2 tickers from a sector are present in the
    window, that factor is omitted.

    Parameters
    ----------
    returns:
        DataFrame of daily returns for the valid assets in the window.
        Columns are ticker strings.

    Returns
    -------
    DataFrame of sector factor returns (columns = factor names present).
    Rows align with returns.index. The Market is NOT included here — it
    is handled as the OLS intercept in estimate_factor_model.
    """
    valid_tickers = set(returns.columns)
    factor_series: Dict[str, pd.Series] = {}

    for sector in ["Semiconductors", "Cloud", "Software"]:
        members = [t for t in SECTOR_MEMBERS[sector] if t in valid_tickers]
        if len(members) >= 2:
            factor_series[sector] = returns[members].mean(axis=1)

    if not factor_series:
        # Fallback: use equal-weight market as the single factor
        factor_series["Market"] = returns.mean(axis=1)

    factor_df = pd.DataFrame(factor_series, index=returns.index)
    return factor_df


def _ols_loadings(
    asset_returns: np.ndarray,   # T-vector
    factor_returns: np.ndarray,  # T x K matrix
) -> Tuple[np.ndarray, float]:
    """OLS regression of one asset on factors.

    Returns loadings (K-vector) and idiosyncratic variance (scalar).
    """
    T, K = factor_returns.shape
    # Add intercept column
    X = np.column_stack([np.ones(T), factor_returns])   # T x (K+1)
    # OLS via normal equations
    coef, _, _, _ = np.linalg.lstsq(X, asset_returns, rcond=None)
    # coef[0] = intercept, coef[1:] = factor loadings
    fitted = X @ coef
    residuals = asset_returns - fitted
    idio_var = float(np.var(residuals, ddof=K + 1))
    return coef[1:], idio_var


def estimate_factor_model(
    returns: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Estimate factor model via OLS for all assets.

    Parameters
    ----------
    returns:
        DataFrame of daily returns (T rows, n columns).

    Returns
    -------
    B : ndarray (n x K)
        Factor loading matrix.
    F_cov : ndarray (K x K)
        Covariance matrix of factor returns (annualized).
    D : ndarray (n,)
        Idiosyncratic variances (annualized). Diagonal of D matrix.
    factor_df : DataFrame
        Factor return series used in estimation.
    """
    factor_df = build_factor_returns(returns)
    K = factor_df.shape[1]
    n = returns.shape[1]
    T = returns.shape[0]

    F = factor_df.values       # T x K
    R = returns.values          # T x n

    B = np.zeros((n, K))
    D_diag = np.zeros(n)

    for i in range(n):
        b_i, idio_var_i = _ols_loadings(R[:, i], F)
        B[i] = b_i
        D_diag[i] = idio_var_i

    # Factor covariance (annualized) with regularization for numerical stability
    # Sector factors can be moderately correlated, causing near-singular F_cov.
    # A small diagonal nudge (1e-6) ensures positive definiteness without
    # materially changing the covariance structure.
    F_cov = np.cov(F.T) * TRADING_DAYS
    if F_cov.ndim == 0:            # single factor edge case
        F_cov = np.array([[float(F_cov)]])
    F_cov += np.eye(F_cov.shape[0]) * 1e-6

    # Annualize idiosyncratic variances
    D_diag_ann = D_diag * TRADING_DAYS

    return B, F_cov, D_diag_ann, factor_df


def factor_covariance_matrix(
    returns: pd.DataFrame,
    shrink_idio: bool = True,
    shrink_factor: float = 0.1,
) -> np.ndarray:
    """Compute annualized factor-model covariance matrix.

    Sigma = B @ F_cov @ B.T + D

    where D is diagonal (idiosyncratic variances), optionally shrunk
    toward the cross-sectional mean to avoid overfitting noisy residuals.

    Parameters
    ----------
    returns:
        DataFrame of daily returns (T rows, n columns = valid assets).
    shrink_idio:
        If True, apply James-Stein-style shrinkage on idiosyncratic variances:
        D_shrunk = (1 - shrink_factor) * D_raw + shrink_factor * mean(D_raw)
        This reduces the impact of extreme idiosyncratic variance estimates.
    shrink_factor:
        Shrinkage intensity toward the cross-sectional mean (default 0.1).

    Returns
    -------
    Annualized covariance matrix as ndarray (n x n).
    """
    B, F_cov, D_diag_ann, _ = estimate_factor_model(returns)

    if shrink_idio and len(D_diag_ann) > 1:
        mean_d = np.mean(D_diag_ann)
        D_diag_ann = (1 - shrink_factor) * D_diag_ann + shrink_factor * mean_d

    # Clip negative values (numerical noise)
    D_diag_ann = np.clip(D_diag_ann, 1e-10, None)

    D = np.diag(D_diag_ann)
    Sigma = B @ F_cov @ B.T + D

    # Ensure symmetry and positive definiteness via small regularization
    Sigma = (Sigma + Sigma.T) / 2
    Sigma += np.eye(len(Sigma)) * 1e-8

    return Sigma


def sample_covariance_matrix(
    returns: pd.DataFrame,
) -> np.ndarray:
    """Compute standard annualized sample covariance matrix.

    This is the default method used before factor model integration.

    Parameters
    ----------
    returns:
        DataFrame of daily returns (T rows, n columns).

    Returns
    -------
    Annualized covariance matrix as ndarray (n x n).
    """
    cov = returns.cov().values * TRADING_DAYS
    # Small regularization for numerical stability
    cov += np.eye(cov.shape[0]) * 1e-8
    return cov


def ledoit_wolf_covariance_matrix(
    returns: pd.DataFrame,
) -> np.ndarray:
    """Compute annualized Ledoit-Wolf shrinkage covariance matrix.

    Ledoit-Wolf (2004) provides an analytical shrinkage estimator that
    minimises the expected Frobenius norm between the estimated and true
    covariance matrices. The shrinkage target is a scaled identity matrix
    and the optimal shrinkage intensity is computed analytically.

    Advantages over sample covariance:
    - Always well-conditioned (positive definite by construction)
    - Reduces estimation error in finite samples
    - Particularly effective when T/n is moderate (< 10-15)

    In this project (T=252, n=17-20): T/n ≈ 13-15, a regime where
    Ledoit-Wolf provides measurable regularisation.

    Parameters
    ----------
    returns:
        DataFrame of daily returns (T rows, n columns).

    Returns
    -------
    Annualized covariance matrix as ndarray (n x n).
    """
    lw = LedoitWolf(assume_centered=False)
    lw.fit(returns.values)
    cov = lw.covariance_ * TRADING_DAYS
    cov += np.eye(cov.shape[0]) * 1e-8
    return cov


def build_covariance_matrix(
    returns: pd.DataFrame,
    method: str = "sample",
) -> np.ndarray:
    """Unified entry point for covariance estimation.

    This is the only function called by the rebalancer.

    Parameters
    ----------
    returns:
        DataFrame of daily returns for valid assets in the current window.
    method:
        "sample"      → standard sample covariance (default, original behavior)
        "factor"      → factor-model covariance via OLS (Barra-style, Σ = BFBᵀ + D)
        "ledoit_wolf" → Ledoit-Wolf analytical shrinkage estimator

    Returns
    -------
    Annualized covariance matrix as ndarray (n x n).

    Raises
    ------
    ValueError if method is not one of the three supported values.
    """
    if method == "sample":
        return sample_covariance_matrix(returns)
    elif method == "factor":
        return factor_covariance_matrix(returns)
    elif method == "ledoit_wolf":
        return ledoit_wolf_covariance_matrix(returns)
    else:
        raise ValueError(
            f"Unknown covariance_method: '{method}'. "
            "Expected 'sample', 'factor', or 'ledoit_wolf'."
        )


def compare_covariance_methods(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Diagnostic: compare sample vs factor covariance properties.

    Useful for understanding the difference between methods on a given window.

    Returns
    -------
    DataFrame with summary statistics for each method.
    """
    cov_sample = sample_covariance_matrix(returns)
    cov_factor = factor_covariance_matrix(returns)

    def _stats(cov: np.ndarray, name: str) -> dict:
        eigvals = np.linalg.eigvalsh(cov)
        return {
            "Method": name,
            "Condition Number": float(np.linalg.cond(cov)),
            "Min Eigenvalue": float(eigvals.min()),
            "Max Eigenvalue": float(eigvals.max()),
            "Mean Diag (Ann. Var)": float(np.diag(cov).mean()),
            "Off-Diag Sparsity": float((np.abs(cov - np.diag(np.diag(cov))) < 1e-6).mean()),
        }

    rows = [_stats(cov_sample, "sample"), _stats(cov_factor, "factor")]
    return pd.DataFrame(rows).set_index("Method")
