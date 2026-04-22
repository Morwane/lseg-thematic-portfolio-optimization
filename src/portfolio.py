"""Portfolio optimization module for computing optimal weights."""

from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

TRADING_DAYS = 252


def equal_weight_portfolio(n_assets: int) -> np.ndarray:
    """Return equal weights array summing to 1.0."""
    return np.array([1.0 / n_assets] * n_assets)


def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """Compute weighted portfolio return from mean returns."""
    return float(np.dot(weights, mean_returns))


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Compute portfolio volatility from covariance matrix."""
    variance = float(weights @ cov_matrix @ weights)
    variance = max(variance, 0.0)
    return float(np.sqrt(variance))


def negative_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> float:
    """Compute negative Sharpe ratio for minimization."""
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    if vol == 0:
        return 1e6
    sharpe = (ret - risk_free_rate) / vol
    return -sharpe


def risk_contribution(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """Compute total risk contribution for each asset."""
    portfolio_vol = portfolio_volatility(weights, cov_matrix)
    if portfolio_vol == 0:
        return np.zeros_like(weights)
    marginal_contrib = cov_matrix @ weights
    risk_contrib = weights * marginal_contrib / portfolio_vol
    return risk_contrib


def erc_objective(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Compute ERC objective: sum of squared deviations from average risk contribution."""
    risk_contrib = risk_contribution(weights, cov_matrix)
    avg_risk = np.mean(risk_contrib)
    objective = np.sum((risk_contrib - avg_risk) ** 2)
    return float(objective)


def min_variance_portfolio(cov_matrix: np.ndarray, max_weight: float) -> np.ndarray:
    """Optimize for minimum variance portfolio with SLSQP and max weight constraint."""
    n_assets = cov_matrix.shape[0]
    initial_weights = equal_weight_portfolio(n_assets)
    
    cov_matrix = cov_matrix + np.eye(n_assets) * 1e-8
    
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    
    result = minimize(
        fun=lambda w: portfolio_volatility(w, cov_matrix),
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    
    if not result.success:
        return equal_weight_portfolio(n_assets)
    
    weights = result.x
    weights = np.clip(weights, 0.0, None)       # clear numerical noise (e.g. -1e-15)
    weights /= weights.sum()                     # renormalise to sum=1
    return weights


def max_sharpe_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    max_weight: float,
) -> np.ndarray:
    """Optimize for maximum Sharpe ratio portfolio with SLSQP and max weight constraint."""
    n_assets = cov_matrix.shape[0]
    initial_weights = equal_weight_portfolio(n_assets)
    
    cov_matrix = cov_matrix + np.eye(n_assets) * 1e-8
    
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    
    result = minimize(
        fun=lambda w: negative_sharpe_ratio(w, mean_returns, cov_matrix, risk_free_rate),
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    
    if not result.success:
        return equal_weight_portfolio(n_assets)
    
    weights = result.x
    weights = np.clip(weights, 0.0, None)       # clear numerical noise (e.g. -1e-15)
    weights /= weights.sum()                     # renormalise to sum=1
    return weights


def equal_risk_contribution_portfolio(cov_matrix: np.ndarray, max_weight: float) -> np.ndarray:
    """Optimize for equal risk contribution portfolio with SLSQP and max weight constraint."""
    n_assets = cov_matrix.shape[0]
    initial_weights = equal_weight_portfolio(n_assets)
    
    cov_matrix = cov_matrix + np.eye(n_assets) * 1e-8
    
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    
    result = minimize(
        fun=lambda w: erc_objective(w, cov_matrix),
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    
    if not result.success:
        return equal_weight_portfolio(n_assets)
    
    weights = result.x
    weights = np.clip(weights, 0.0, None)       # clear numerical noise (e.g. -1e-15)
    weights /= weights.sum()                     # renormalise to sum=1
    return weights


def compute_historical_cvar(
    portfolio_returns: np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """
    Compute CVaR (Conditional Value at Risk) from historical portfolio returns.
    
    CVaR = expected value of returns in the worst (1-confidence_level)% tail.
    
    Args:
        portfolio_returns: 1D array of historical portfolio returns.
        confidence_level: Confidence level (e.g., 0.95 for 95% CVaR).
    
    Returns:
        CVaR value (negative for losses, positive for gains in tail).
    """
    var_percentile = (1.0 - confidence_level) * 100
    var = np.percentile(portfolio_returns, var_percentile)
    # Include returns equal to VaR in the tail average
    tail_returns = portfolio_returns[portfolio_returns <= var]
    if len(tail_returns) == 0:
        return float(np.min(portfolio_returns))
    cvar = float(np.mean(tail_returns))
    return cvar


def min_cvar_portfolio(
    returns: pd.DataFrame,
    max_weight: float,
    confidence_level: float = 0.95,
) -> np.ndarray:
    """Minimize CVaR via the exact Rockafellar-Uryasev LP reformulation.

    This is the industry-standard exact formulation for CVaR minimization.
    It avoids the local-minima problem that SLSQP encounters on the
    non-smooth CVaR objective, which causes momentum-concentrated portfolios
    instead of genuinely risk-minimizing ones.

    LP formulation (Rockafellar & Uryasev, 2000):

        min_{w, zeta, u}  zeta + 1/((1-alpha)*T) * sum_t(u_t)

        s.t.  u_t >= -(r_t @ w) - zeta    for all t=1..T   [tail loss slack]
              u_t >= 0                      for all t
              sum(w) = 1                    [fully invested]
              0 <= w_i <= max_weight        [long-only with cap]
              zeta free                     [VaR at alpha level]

    where alpha = confidence_level (e.g. 0.95), T = number of periods,
    r_t = return vector on day t, zeta = VaR threshold.

    Parameters
    ----------
    returns:
        DataFrame of asset daily returns (rows=periods, columns=assets).
    max_weight:
        Maximum weight per asset (e.g. 0.20 for 20% cap).
    confidence_level:
        CVaR confidence level (default 0.95 = 95% CVaR / Expected Shortfall).

    Returns
    -------
    np.ndarray of optimal weights summing to 1.
    """
    R = returns.values          # T × n
    T, n = R.shape
    alpha = confidence_level
    scale = 1.0 / ((1.0 - alpha) * T)

    # Variable layout: [w_0..w_{n-1}, zeta, u_0..u_{T-1}]
    # Objective: 0*w + 1*zeta + scale*u
    c = np.zeros(n + 1 + T)
    c[n] = 1.0
    c[n + 1:] = scale

    # Inequality constraints: u_t + (r_t @ w) + zeta >= 0
    # => -(r_t @ w) - zeta - u_t <= 0
    # A_ub @ x <= b_ub  with  A_ub[t, :n] = -R[t], A_ub[t, n] = -1, A_ub[t, n+1+t] = -1
    A_ub = np.zeros((T, n + 1 + T))
    A_ub[:, :n] = -R
    A_ub[:, n] = -1.0
    for t in range(T):
        A_ub[t, n + 1 + t] = -1.0
    b_ub = np.zeros(T)

    # Equality: sum(w) = 1
    A_eq = np.zeros((1, n + 1 + T))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # Bounds: 0 <= w_i <= max_weight, zeta free, u_t >= 0
    bounds = (
        [(0.0, max_weight)] * n
        + [(None, None)]        # zeta
        + [(0.0, None)] * T     # u_t
    )

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return equal_weight_portfolio(n)

    weights = result.x[:n]
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0:
        return equal_weight_portfolio(n)
    return weights / total


def weights_to_series(weights: np.ndarray, tickers: List[str]) -> pd.Series:
    """Convert weight array to Series indexed by ticker."""
    return pd.Series(weights, index=tickers)
