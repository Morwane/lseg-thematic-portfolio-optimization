"""Rebalancer module for walk-forward portfolio optimization with monthly rebalancing."""

import pandas as pd
import numpy as np

from src.covariance import build_covariance_matrix
from src.metrics import portfolio_turnover, transaction_cost_from_turnover


def generate_rebalance_dates(
    returns: pd.DataFrame,
    frequency: str = "ME",
) -> pd.DatetimeIndex:
    """Generate rebalance dates from returns index using specified frequency."""
    rebalance_dates = returns.resample(frequency).last().index
    return pd.DatetimeIndex(rebalance_dates)


def apply_weights(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Apply weights to returns and compute daily portfolio returns.

    Only uses assets with non-zero weights to avoid NaN contamination
    from assets that are not yet in the optimizer universe.
    """
    active_assets = weights[weights > 1e-10].index.tolist()
    aligned_returns = returns[active_assets]
    active_weights = weights[active_assets]
    weighted_returns = aligned_returns.multiply(active_weights.values, axis=1)
    portfolio_daily_returns = weighted_returns.sum(axis=1)
    return portfolio_daily_returns


def _clean_weights(weights_array: np.ndarray) -> np.ndarray:
    """Zero out near-zero weights (numerical noise) and re-normalize to sum=1."""
    w = np.array(weights_array, dtype=float)
    w[w < 1e-8] = 0.0
    total = w.sum()
    if total <= 0:
        return np.ones(len(w)) / len(w)
    return w / total


def walk_forward_rebalance(
    returns: pd.DataFrame,
    optimizer_func,
    optimizer_name: str,
    lookback_window_days: int,
    rebalance_frequency: str,
    transaction_cost_bps: float,
    max_weight: float,
    risk_free_rate: float = 0.0,
    covariance_method: str = "sample",
    bl_risk_aversion: float = 2.5,
    bl_tau: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Execute walk-forward rebalancing with per-window valid-asset filtering.

    For each rebalance date:
    - Trains only on assets with complete history in the lookback window (dropna).
    - Optimizes on the valid sub-universe.
    - Reconstructs full-universe weights (zero for excluded assets).
    - Tracks fallback usage in metadata.

    Parameters
    ----------
    returns:
        Full daily returns DataFrame for the entire universe.
    optimizer_func:
        Portfolio optimization function.
    optimizer_name:
        String key identifying the optimizer (controls dispatch logic).
    lookback_window_days:
        Number of trading days in each training window.
    rebalance_frequency:
        Pandas offset alias (e.g. "ME" for month-end).
    transaction_cost_bps:
        Transaction cost in basis points applied on first day of each period.
    max_weight:
        Maximum weight per asset (long-only cap).
    risk_free_rate:
        Annualized risk-free rate for Sharpe-based optimizers.
    covariance_method:
        "sample"      → standard sample covariance (default, original behavior).
        "factor"      → factor-model covariance via OLS (Barra-style).
        "ledoit_wolf" → Ledoit-Wolf analytical shrinkage.
    bl_risk_aversion:
        Black-Litterman lambda parameter (only used when optimizer_name="black_litterman").
    bl_tau:
        Black-Litterman tau parameter (only used when optimizer_name="black_litterman").

    Returns
    -------
    full_portfolio_returns : pd.Series
        Daily portfolio returns across the full backtest period.
    weights_history_df : pd.DataFrame
        One row per rebalance date, columns = full ticker universe.
    meta_df : pd.DataFrame
        Per-window metadata: rebalance_date, optimizer_name,
        n_valid_assets, used_equal_weight_fallback, covariance_method.
    """
    if len(returns) <= lookback_window_days:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    rebalance_dates = generate_rebalance_dates(returns, rebalance_frequency)
    rebalance_dates = rebalance_dates[rebalance_dates >= returns.index[lookback_window_days]]

    portfolio_return_blocks: list[pd.Series] = []
    weights_history_list: list[pd.Series] = []
    optimization_meta: list[dict] = []
    previous_weights = pd.Series(0.0, index=returns.columns, dtype=float)

    for i, rebalance_date in enumerate(rebalance_dates):
        rebalance_idx = returns.index.get_indexer([rebalance_date], method="nearest")[0]

        if rebalance_idx < lookback_window_days:
            continue

        train_start_idx = rebalance_idx - lookback_window_days
        train_slice = returns.iloc[train_start_idx:rebalance_idx]

        # --- Filter: keep only assets with NO missing values in this window ---
        valid_train_slice = train_slice.dropna(axis=1)

        if len(valid_train_slice.columns) < 2:
            optimization_meta.append({
                "rebalance_date": rebalance_date,
                "optimizer_name": optimizer_name,
                "n_valid_assets": len(valid_train_slice.columns),
                "used_equal_weight_fallback": True,
            })
            continue

        valid_assets = valid_train_slice.columns.tolist()
        n_valid = len(valid_assets)

        mean_returns = valid_train_slice.mean().values * 252
        cov_matrix = build_covariance_matrix(valid_train_slice, method=covariance_method)

        used_fallback = False

        # Guard: max_weight must allow a feasible portfolio
        if n_valid * max_weight < 1.0:
            weights_array = np.ones(n_valid) / n_valid
            used_fallback = True
        else:
            try:
                if optimizer_name == "equal_weight":
                    weights_array = optimizer_func(n_valid)
                elif optimizer_name == "min_variance":
                    weights_array = optimizer_func(cov_matrix, max_weight)
                elif optimizer_name == "max_sharpe":
                    weights_array = optimizer_func(mean_returns, cov_matrix, risk_free_rate, max_weight)
                elif optimizer_name == "erc":
                    weights_array = optimizer_func(cov_matrix, max_weight)
                elif optimizer_name in ["min_cvar", "min_cvar_uncapped"]:
                    # CVaR optimizers use historical returns directly
                    weights_array = optimizer_func(valid_train_slice, max_weight)
                elif optimizer_name == "black_litterman":
                    # BL walk-forward: momentum views generated from training window
                    # optimizer_func is not used — BL logic is self-contained
                    from src.black_litterman import black_litterman_optimizer
                    weights_array = black_litterman_optimizer(
                        valid_train_slice=valid_train_slice,
                        cov_matrix=cov_matrix,
                        mean_returns=mean_returns,
                        valid_assets=valid_assets,
                        risk_free_rate=risk_free_rate,
                        max_weight=max_weight,
                        risk_aversion=bl_risk_aversion,
                        tau=bl_tau,
                    )
                else:
                    raise ValueError(f"Unknown optimizer_name: {optimizer_name}")
            except Exception:
                weights_array = np.ones(n_valid) / n_valid
                used_fallback = True

        # Clean numerical noise and re-normalize
        weights_array = _clean_weights(weights_array)

        # Rebuild full-universe weights (zero for assets not in valid set)
        current_weights = pd.Series(0.0, index=returns.columns, dtype=float)
        current_weights.loc[valid_assets] = weights_array

        # Clean up near-zero weights and re-normalize
        current_weights[current_weights < 1e-6] = 0.0
        total = current_weights.sum()
        if total > 0:
            current_weights = current_weights / total

        current_weights.name = rebalance_date

        optimization_meta.append({
            "rebalance_date": rebalance_date,
            "optimizer_name": optimizer_name,
            "n_valid_assets": n_valid,
            "used_equal_weight_fallback": used_fallback,
            "covariance_method": covariance_method,
        })

        # Transaction costs applied on first day of the new period
        turnover = portfolio_turnover(previous_weights, current_weights)
        transaction_cost = transaction_cost_from_turnover(turnover, transaction_cost_bps)

        # Determine holding period: from this rebalance to the next
        if i + 1 < len(rebalance_dates):
            next_rebalance_date = rebalance_dates[i + 1]
            next_rebalance_idx = returns.index.get_indexer([next_rebalance_date], method="nearest")[0]
        else:
            next_rebalance_idx = len(returns)

        out_of_sample_slice = returns.iloc[rebalance_idx + 1:next_rebalance_idx]

        block_portfolio_returns = apply_weights(out_of_sample_slice, current_weights)

        if block_portfolio_returns.empty:
            continue

        # Deduct transaction cost on the first day of the holding period
        block_portfolio_returns.iloc[0] = block_portfolio_returns.iloc[0] - transaction_cost

        portfolio_return_blocks.append(block_portfolio_returns)
        weights_history_list.append(current_weights)
        previous_weights = current_weights

    # --- Assemble outputs ---
    meta_df = pd.DataFrame(optimization_meta)

    if not portfolio_return_blocks:
        return pd.Series(dtype=float), pd.DataFrame(), meta_df

    full_portfolio_returns = (
        pd.concat(portfolio_return_blocks)
        .sort_index()
    )
    full_portfolio_returns = full_portfolio_returns[
        ~full_portfolio_returns.index.duplicated(keep="first")
    ]

    weights_history_df = pd.DataFrame(weights_history_list)
    if not weights_history_df.empty:
        weights_history_df.index = pd.to_datetime([w.name for w in weights_history_list])

    return full_portfolio_returns, weights_history_df, meta_df
