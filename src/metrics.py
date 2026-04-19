"""Metrics module for computing risk and performance metrics."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    """Compute annualized return from daily returns."""
    if len(returns) == 0:
        return float(np.nan)
    compounded = (1 + returns).prod()
    annualized = compounded ** (TRADING_DAYS / len(returns)) - 1
    return float(annualized)


def annualized_volatility(returns: pd.Series) -> float:
    """Compute annualized volatility from daily returns."""
    if len(returns) == 0:
        return float(np.nan)
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute Sharpe ratio from daily returns and risk-free rate."""
    ann_return = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    if ann_vol == 0 or np.isnan(ann_vol):
        return float(np.nan)
    return (ann_return - risk_free_rate) / ann_vol


def drawdown_series(cumulative_returns: pd.Series) -> pd.Series:
    """Compute drawdown series from cumulative returns."""
    if len(cumulative_returns) == 0:
        return pd.Series(dtype=float)
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1
    return drawdown


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Compute maximum drawdown from cumulative returns."""
    drawdown = drawdown_series(cumulative_returns)
    if len(drawdown) == 0:
        return float(np.nan)
    return float(drawdown.min())


def portfolio_turnover(old_weights: pd.Series, new_weights: pd.Series) -> float:
    """Compute portfolio turnover as sum of absolute weight changes."""
    old_aligned = old_weights.reindex(new_weights.index, fill_value=0.0)
    weight_diff = (new_weights - old_aligned).abs()
    turnover = weight_diff.sum()
    return float(turnover)


def transaction_cost_from_turnover(turnover: float, transaction_cost_bps: float) -> float:
    """Compute transaction cost from turnover and basis points cost."""
    cost = turnover * (transaction_cost_bps / 10000.0)
    return float(cost)


def rolling_sharpe(returns: pd.Series, window: int = 60, risk_free_rate: float = 0.0) -> pd.Series:
    """Compute rolling Sharpe ratio from daily returns over specified window."""
    rolling_mean = returns.rolling(window=window).mean() * TRADING_DAYS
    rolling_std = returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS)
    rolling_sharpe_values = (rolling_mean - risk_free_rate) / rolling_std
    return rolling_sharpe_values


def rolling_volatility(returns: pd.Series, window: int = 60) -> pd.Series:
    """Compute rolling volatility from daily returns over specified window."""
    rolling_std = returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS)
    return rolling_std
