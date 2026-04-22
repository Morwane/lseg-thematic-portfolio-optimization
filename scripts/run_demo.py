"""
Demo mode — runs the full portfolio optimization pipeline on synthetic data.

No LSEG credentials or local LSEG Desktop are required.
All outputs go to output/demo/ and are clearly labelled as synthetic.

Usage:
    python scripts/run_demo.py

What this produces:
    output/demo/charts/   — cumulative performance, drawdowns, rolling metrics,
                            ERC flagship charts (capital vs risk, weight stability)
    output/demo/reports/  — portfolio_summary.csv, erc_weights_history.csv, ...

The synthetic universe mimics the volatility structure of the real AI/Tech
universe (high-vol semiconductors, lower-vol cloud/enterprise names) without
using any real prices. Results are illustrative only.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path regardless of where the script is invoked from
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

# --- project imports --------------------------------------------------------
from src.backtest import cumulative_performance
from src.metrics import (
    annualized_return,
    annualized_volatility,
    drawdown_series,
    max_drawdown,
    rolling_sharpe,
    rolling_volatility,
    sharpe_ratio,
)
from src.portfolio import (
    TRADING_DAYS,
    equal_risk_contribution_portfolio,
    equal_weight_portfolio,
    max_sharpe_portfolio,
    min_cvar_portfolio,
    min_variance_portfolio,
    risk_contribution,
)
from src.rebalancer import walk_forward_rebalance
from src.visualization import (
    plot_cumulative_performance,
    plot_drawdown_series,
    plot_erc_vs_ew_capital_vs_risk,
    plot_erc_weight_stability,
    plot_rolling_sharpe,
    plot_rolling_volatility,
)

# ---------------------------------------------------------------------------
# Synthetic universe — 20 tickers with realistic vol tiers
# ---------------------------------------------------------------------------

DEMO_TICKERS = [
    # High-vol semiconductors (ann. vol ~55–65%)
    "SEMI_1", "SEMI_2", "SEMI_3", "SEMI_4", "SEMI_5",
    # Mid-vol growth (ann. vol ~35–45%)
    "GROW_1", "GROW_2", "GROW_3", "GROW_4", "GROW_5",
    # Lower-vol cloud/enterprise (ann. vol ~20–30%)
    "CLOU_1", "CLOU_2", "CLOU_3", "CLOU_4", "CLOU_5",
    # Defensive enterprise (ann. vol ~15–22%)
    "DEFN_1", "DEFN_2", "DEFN_3", "DEFN_4", "DEFN_5",
]

# Annualised volatility targets per group (daily = ann / sqrt(252))
_VOL_ANN = [
    0.60, 0.58, 0.62, 0.55, 0.57,   # high-vol semis
    0.40, 0.38, 0.42, 0.36, 0.44,   # mid-vol growth
    0.25, 0.22, 0.28, 0.24, 0.26,   # cloud/enterprise
    0.18, 0.20, 0.17, 0.19, 0.21,   # defensive
]

# Annualised drift per group (slightly positive to represent equity premium)
_MU_ANN = [
    0.22, 0.20, 0.24, 0.18, 0.21,   # high-vol semis
    0.16, 0.15, 0.17, 0.14, 0.18,   # mid-vol growth
    0.11, 0.10, 0.12, 0.10, 0.11,   # cloud/enterprise
    0.08, 0.09, 0.08, 0.09, 0.10,   # defensive
]


def generate_synthetic_returns(
    tickers: list[str],
    vols_ann: list[float],
    mus_ann: list[float],
    n_days: int = 1320,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate correlated daily returns with a realistic factor structure.

    Uses a 3-factor model (market + growth + defensive) plus idiosyncratic noise
    so that within-group correlations are higher than cross-group correlations.
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)
    vols_d = np.array(vols_ann) / np.sqrt(TRADING_DAYS)
    mus_d = np.array(mus_ann) / TRADING_DAYS

    # Factor loadings: all assets load on market; group-specific factor loadings
    n_groups = 4
    group_size = n // n_groups
    market_beta = rng.uniform(0.6, 1.1, n)
    group_beta = np.zeros((n, n_groups))
    for g in range(n_groups):
        idx = slice(g * group_size, (g + 1) * group_size)
        group_beta[idx, g] = rng.uniform(0.3, 0.6, group_size)

    factor_vols_d = np.array([0.010, 0.006, 0.006, 0.006, 0.006])  # market + 4 groups

    # Daily factor returns
    n_factors = 1 + n_groups
    factor_returns = rng.normal(0, 1, (n_days, n_factors)) * factor_vols_d

    # Systematic part of each asset's return
    loadings = np.column_stack([market_beta] + [group_beta[:, g] for g in range(n_groups)])
    systematic = factor_returns @ loadings.T

    # Idiosyncratic noise scaled so total vol matches target
    systematic_var = (loadings ** 2 * factor_vols_d ** 2).sum(axis=1)
    idio_vol = np.sqrt(np.maximum(vols_d ** 2 - systematic_var, (vols_d * 0.3) ** 2))
    idio = rng.normal(0, 1, (n_days, n)) * idio_vol

    returns_raw = systematic + idio + mus_d
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    return pd.DataFrame(returns_raw, index=dates, columns=tickers)


def _calmar(ret_series: pd.Series, cumul_series: pd.Series) -> float:
    ann_ret = annualized_return(ret_series)
    dd = max_drawdown(cumul_series)
    return ann_ret / abs(dd) if abs(dd) > 1e-8 else float("nan")


def run_demo() -> None:
    """Run the full portfolio optimization demo on synthetic data."""
    print("=" * 65)
    print("  DEMO MODE — synthetic AI/Tech-like universe (no LSEG required)")
    print("=" * 65)
    print()
    print("Generating synthetic returns for 20-stock universe...")

    returns = generate_synthetic_returns(
        tickers=DEMO_TICKERS,
        vols_ann=_VOL_ANN,
        mus_ann=_MU_ANN,
    )
    print(f"  {len(returns)} trading days | {len(returns.columns)} assets")
    print(f"  Period: {returns.index[0].date()} → {returns.index[-1].date()}")

    # --- Walk-forward rebalancing ---
    lookback = 252
    rebalance_freq = "ME"
    tx_cost = 15.0
    max_w = 0.20
    rf = 0.04
    cov_method = "sample"

    print("\nRunning walk-forward strategies...")

    ew_ret, ew_wh, _ = walk_forward_rebalance(
        returns=returns, optimizer_func=equal_weight_portfolio,
        optimizer_name="equal_weight", lookback_window_days=lookback,
        rebalance_frequency=rebalance_freq, transaction_cost_bps=tx_cost,
        max_weight=max_w, risk_free_rate=rf, covariance_method=cov_method,
    )
    mv_ret, mv_wh, _ = walk_forward_rebalance(
        returns=returns, optimizer_func=min_variance_portfolio,
        optimizer_name="min_variance", lookback_window_days=lookback,
        rebalance_frequency=rebalance_freq, transaction_cost_bps=tx_cost,
        max_weight=max_w, risk_free_rate=rf, covariance_method=cov_method,
    )
    ms_ret, ms_wh, _ = walk_forward_rebalance(
        returns=returns, optimizer_func=max_sharpe_portfolio,
        optimizer_name="max_sharpe", lookback_window_days=lookback,
        rebalance_frequency=rebalance_freq, transaction_cost_bps=tx_cost,
        max_weight=max_w, risk_free_rate=rf, covariance_method=cov_method,
    )
    erc_ret, erc_wh, _ = walk_forward_rebalance(
        returns=returns, optimizer_func=equal_risk_contribution_portfolio,
        optimizer_name="erc", lookback_window_days=lookback,
        rebalance_frequency=rebalance_freq, transaction_cost_bps=tx_cost,
        max_weight=max_w, risk_free_rate=rf, covariance_method=cov_method,
    )
    print("  Computing Min CVaR (LP)...")
    cvar_ret, cvar_wh, _ = walk_forward_rebalance(
        returns=returns, optimizer_func=min_cvar_portfolio,
        optimizer_name="min_cvar", lookback_window_days=lookback,
        rebalance_frequency=rebalance_freq, transaction_cost_bps=tx_cost,
        max_weight=max_w, risk_free_rate=rf, covariance_method=cov_method,
    )

    # --- Cumulative performance and drawdowns ---
    ew_cumul = cumulative_performance(ew_ret.copy())
    mv_cumul = cumulative_performance(mv_ret.copy())
    ms_cumul = cumulative_performance(ms_ret.copy())
    erc_cumul = cumulative_performance(erc_ret.copy())
    cvar_cumul = cumulative_performance(cvar_ret.copy())

    cumul_df = pd.DataFrame({
        "Equal Weight": ew_cumul, "Min Variance": mv_cumul,
        "Max Sharpe": ms_cumul, "ERC": erc_cumul, "Min CVaR": cvar_cumul,
    })
    drawdown_df = pd.DataFrame({
        "Equal Weight": drawdown_series(ew_cumul),
        "Min Variance": drawdown_series(mv_cumul),
        "Max Sharpe": drawdown_series(ms_cumul),
        "ERC": drawdown_series(erc_cumul),
        "Min CVaR": drawdown_series(cvar_cumul),
    })

    # --- Rolling metrics ---
    rolling_w = 252
    rolling_sharpe_df = pd.DataFrame({
        "Equal Weight": rolling_sharpe(ew_ret, window=rolling_w, risk_free_rate=rf),
        "Min Variance": rolling_sharpe(mv_ret, window=rolling_w, risk_free_rate=rf),
        "Max Sharpe": rolling_sharpe(ms_ret, window=rolling_w, risk_free_rate=rf),
        "ERC": rolling_sharpe(erc_ret, window=rolling_w, risk_free_rate=rf),
        "Min CVaR": rolling_sharpe(cvar_ret, window=rolling_w, risk_free_rate=rf),
    })
    rolling_vol_df = pd.DataFrame({
        "Equal Weight": rolling_volatility(ew_ret, window=rolling_w),
        "Min Variance": rolling_volatility(mv_ret, window=rolling_w),
        "Max Sharpe": rolling_volatility(ms_ret, window=rolling_w),
        "ERC": rolling_volatility(erc_ret, window=rolling_w),
        "Min CVaR": rolling_volatility(cvar_ret, window=rolling_w),
    })

    # --- Summary table ---
    summary = pd.DataFrame({
        "Portfolio": ["Equal Weight", "Min Variance", "Max Sharpe", "ERC", "Min CVaR"],
        "Ann. Return": [
            annualized_return(ew_ret), annualized_return(mv_ret),
            annualized_return(ms_ret), annualized_return(erc_ret),
            annualized_return(cvar_ret),
        ],
        "Ann. Vol": [
            annualized_volatility(ew_ret), annualized_volatility(mv_ret),
            annualized_volatility(ms_ret), annualized_volatility(erc_ret),
            annualized_volatility(cvar_ret),
        ],
        "Sharpe": [
            sharpe_ratio(ew_ret, rf), sharpe_ratio(mv_ret, rf),
            sharpe_ratio(ms_ret, rf), sharpe_ratio(erc_ret, rf),
            sharpe_ratio(cvar_ret, rf),
        ],
        "Max DD": [
            max_drawdown(ew_cumul), max_drawdown(mv_cumul),
            max_drawdown(ms_cumul), max_drawdown(erc_cumul),
            max_drawdown(cvar_cumul),
        ],
        "Calmar": [
            _calmar(ew_ret, ew_cumul), _calmar(mv_ret, mv_cumul),
            _calmar(ms_ret, ms_cumul), _calmar(erc_ret, erc_cumul),
            _calmar(cvar_ret, cvar_cumul),
        ],
    }).set_index("Portfolio")

    print("\n=== DEMO — Strategy Summary (synthetic data) ===\n")
    print(summary.round(4).to_string())

    # --- ERC risk contribution analysis ---
    print("\n=== ERC Risk Contribution Analysis ===")
    erc_final = erc_wh.iloc[-1].reindex(DEMO_TICKERS).fillna(0.0)
    ew_final = ew_wh.iloc[-1].reindex(DEMO_TICKERS).fillna(0.0)
    cov_full = returns.cov().values * TRADING_DAYS

    erc_rc = risk_contribution(erc_final.values, cov_full)
    ew_rc = risk_contribution(ew_final.values, cov_full)

    print(f"  ERC max risk contribution deviation from equal: "
          f"{np.abs(erc_rc - erc_rc.mean()).max():.4f}")
    print(f"  EW max risk contribution deviation from equal:  "
          f"{np.abs(ew_rc - ew_rc.mean()).max():.4f}")
    print(f"  → ERC reduces max RC deviation by "
          f"{(1 - np.abs(erc_rc - erc_rc.mean()).max() / np.abs(ew_rc - ew_rc.mean()).max()):.0%}")

    # --- Output ---
    out_charts = Path("output/demo/charts")
    out_reports = Path("output/demo/reports")
    out_charts.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    summary.to_csv(out_reports / "portfolio_summary_demo.csv")
    cumul_df.to_csv(out_reports / "cumulative_performance_demo.csv")
    erc_wh.to_csv(out_reports / "erc_weights_history_demo.csv")

    # Save charts
    plot_cumulative_performance(
        cumul_df, output_path=str(out_charts / "cumulative_performance_demo.png"))
    plot_drawdown_series(
        drawdown_df, output_path=str(out_charts / "drawdowns_demo.png"))
    plot_rolling_sharpe(
        rolling_sharpe_df, output_path=str(out_charts / "rolling_sharpe_demo.png"))
    plot_rolling_volatility(
        rolling_vol_df, output_path=str(out_charts / "rolling_volatility_demo.png"))
    plot_erc_vs_ew_capital_vs_risk(
        erc_weights=erc_final,
        ew_weights=ew_final,
        cov_matrix=cov_full,
        output_path=str(out_charts / "erc_vs_ew_capital_risk_demo.png"),
    )
    plot_erc_weight_stability(
        erc_weights_history=erc_wh,
        output_path=str(out_charts / "erc_weight_stability_demo.png"),
    )

    print(f"\nOutputs saved to {out_charts}/ and {out_reports}/")
    print("\nNOTE: All results above use synthetic data.")
    print("      They illustrate the methodology but do not represent real market results.")
    print("      Run main.py with LSEG credentials for the full live-data pipeline.\n")


if __name__ == "__main__":
    run_demo()
