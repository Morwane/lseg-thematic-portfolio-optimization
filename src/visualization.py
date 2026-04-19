"""Visualization module for producing and saving publication-ready charts."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

STRATEGY_COLORS = {
    "Equal Weight":    "#4C72B0",
    "Min Variance":    "#55A868",
    "Max Sharpe":      "#C44E52",
    "ERC":             "#DD8452",
    "Min CVaR":        "#8172B2",
    "BL Walk-Forward": "#777777",
}


def _pct_formatter(y, _):
    return f"{y:.0%}"


# ---------------------------------------------------------------------------
# Core performance charts
# ---------------------------------------------------------------------------

def plot_cumulative_performance(
    perf_df: pd.DataFrame,
    output_path: str = "output/charts/cumulative_performance.png",
) -> None:
    """Line chart of cumulative portfolio performance (base = 1.0)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    for col in perf_df.columns:
        color = STRATEGY_COLORS.get(col, "#888888")
        ax.plot(perf_df.index, perf_df[col], label=col, linewidth=2, color=color)

    ax.set_title("Cumulative Portfolio Performance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value (base = 1.0)", fontsize=12)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}x"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_drawdown_series(
    drawdown_df: pd.DataFrame,
    output_path: str = "output/charts/drawdowns_v2.png",
) -> None:
    """Shaded area chart of drawdown series — the most readable risk visual."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    for col in drawdown_df.columns:
        color = STRATEGY_COLORS.get(col, "#888888")
        ax.fill_between(drawdown_df.index, drawdown_df[col], 0,
                        alpha=0.18, color=color)
        ax.plot(drawdown_df.index, drawdown_df[col],
                label=col, linewidth=1.6, color=color)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Portfolio Drawdown Series", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown from Peak", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_weights(
    weights_dict: dict,
    output_path: str = "output/charts/portfolio_weights.png",
    max_portfolios: int = 4,
) -> None:
    """Horizontal bar charts of portfolio weights.

    Limits to max_portfolios to keep the figure readable.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Trim to at most max_portfolios strategies
    items = list(weights_dict.items())[:max_portfolios]
    n = len(items)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for idx, (name, weights) in enumerate(items):
        color = STRATEGY_COLORS.get(name, "#4C72B0")
        axes[idx].barh(weights.index, weights.values, color=color, alpha=0.85)
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Weight", fontsize=11)
        axes[idx].xaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
        axes[idx].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_efficient_frontier(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    portfolios: dict,
    risk_free_rate: float,
    output_path: str = "output/charts/efficient_frontier.png",
    n_simulations: int = 3000,
) -> None:
    """Scatter plot of efficient frontier with random portfolios and optimal portfolios."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    n_assets = len(mean_returns)
    sim_returns = np.zeros(n_simulations)
    sim_vols = np.zeros(n_simulations)
    sim_sharpes = np.zeros(n_simulations)

    rng = np.random.default_rng(42)
    for i in range(n_simulations):
        w = rng.dirichlet(np.ones(n_assets))
        sim_returns[i] = w @ mean_returns
        sim_vols[i] = np.sqrt(w @ cov_matrix @ w)
        sim_sharpes[i] = (
            (sim_returns[i] - risk_free_rate) / sim_vols[i] if sim_vols[i] > 0 else 0
        )

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(sim_vols, sim_returns, c=sim_sharpes,
                         cmap="viridis", alpha=0.45, s=20)

    markers = {"Equal Weight": "o", "Min Variance": "s",
               "Max Sharpe": "^", "ERC": "D", "Min CVaR": "P"}
    for name, weights in portfolios.items():
        port_ret = weights @ mean_returns
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        ax.scatter(
            port_vol, port_ret,
            color=STRATEGY_COLORS.get(name, "purple"),
            marker=markers.get(name, "o"),
            s=200, label=name, edgecolors="black", linewidth=1.5, zorder=5,
        )

    ax.set_title("Efficient Frontier", fontsize=14, fontweight="bold")
    ax.set_xlabel("Annualized Volatility", fontsize=12)
    ax.set_ylabel("Annualized Return", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rolling_sharpe(
    rolling_sharpe_df: pd.DataFrame,
    output_path: str = "output/charts/rolling_sharpe.png",
) -> None:
    """Line chart of rolling Sharpe ratio over time."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    for col in rolling_sharpe_df.columns:
        ax.plot(rolling_sharpe_df.index, rolling_sharpe_df[col],
                label=col, linewidth=1.8,
                color=STRATEGY_COLORS.get(col, "#888888"))

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Rolling Sharpe Ratio", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Sharpe Ratio (rolling)", fontsize=12)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rolling_volatility(
    rolling_vol_df: pd.DataFrame,
    output_path: str = "output/charts/rolling_volatility.png",
) -> None:
    """Line chart of rolling annualized volatility over time."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    for col in rolling_vol_df.columns:
        ax.plot(rolling_vol_df.index, rolling_vol_df[col],
                label=col, linewidth=1.8,
                color=STRATEGY_COLORS.get(col, "#888888"))

    ax.set_title("Rolling Volatility (Annualized)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Annualized Volatility", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Stress testing charts
# ---------------------------------------------------------------------------

def plot_stress_comparison(
    stress_df: pd.DataFrame,
    output_path: str = "output/charts/stress_comparison_v2.png",
) -> None:
    """Grouped bar chart of cumulative returns per strategy across stress scenarios.

    Bars are colored by strategy (consistent palette), not by return sign.
    Only scenarios with actual data (N Days > 0) are shown.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Drop rows with no data
    data = stress_df[stress_df["N Days"] > 0]["Cumulative Return"].unstack()

    if data.empty:
        return

    strategies = data.columns.tolist()
    scenarios = data.index.tolist()
    x = np.arange(len(scenarios))
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, strategy in enumerate(strategies):
        offset = (i - len(strategies) / 2 + 0.5) * width
        values = data[strategy].values
        color = STRATEGY_COLORS.get(strategy, "#888888")
        bars = ax.bar(x + offset, values, width, label=strategy,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

        # Annotate each bar with its value
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + (0.005 if val >= 0 else -0.018),
                    f"{val:.1%}",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=8, fontweight="bold",
                )

    ax.axhline(0, color="black", linewidth=1.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=12, ha="right", fontsize=11)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.set_title("Portfolio Performance During Historical Stress Scenarios",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_stress_heatmap(
    stress_df: pd.DataFrame,
    metric: str = "Max Drawdown",
    output_path: str = "output/charts/stress_heatmap_v2_maxdd.png",
) -> None:
    """Heatmap of one stress metric across all (scenario, strategy) combinations.

    Rows with no data (N Days == 0) are shown as blank.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Build matrix, mask no-data rows
    matrix = stress_df[metric].unstack()

    fig, ax = plt.subplots(figsize=(10, max(4, len(matrix) * 1.4)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2%",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": metric, "format": mticker.FuncFormatter(_pct_formatter)},
        ax=ax,
        annot_kws={"size": 10},
    )
    ax.set_title(f"Stress Test Heatmap — {metric}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Strategy", fontsize=11)
    ax.set_ylabel("Scenario", fontsize=11)
    ax.tick_params(axis="x", labelrotation=15)
    ax.tick_params(axis="y", labelrotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
