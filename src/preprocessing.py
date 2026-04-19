"""Preprocessing module for cleaning prices and computing returns."""

import pandas as pd


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Clean price data by sorting, forward filling, and dropping fully missing rows."""
    prices_sorted = prices.sort_index()
    prices_filled = prices_sorted.ffill()
    prices_cleaned = prices_filled.dropna(how="all")
    return prices_cleaned


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from price data."""
    returns = prices.pct_change()
    returns_cleaned = returns.dropna(how="all")
    return returns_cleaned
