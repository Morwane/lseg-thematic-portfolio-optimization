"""Backtesting module for applying weights to returns and computing performance."""

import pandas as pd


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Compute daily portfolio returns by applying weights to return columns."""
    aligned_returns = returns[weights.index]
    weighted_returns = aligned_returns.multiply(weights.values, axis=1)
    portfolio_daily_returns = weighted_returns.sum(axis=1)
    return portfolio_daily_returns


def cumulative_performance(return_series: pd.Series) -> pd.Series:
    """Compute cumulative performance from daily returns."""
    cumulative = (1 + return_series).cumprod()
    return cumulative
