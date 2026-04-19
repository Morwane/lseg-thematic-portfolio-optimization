"""Stress testing module for portfolio robustness analysis during historical crises."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def define_historical_scenarios() -> Dict[str, Tuple[str, str]]:
    """Define historical stress scenarios with fixed date ranges.

    All three scenarios are major drawdown events directly relevant
    to an AI/Tech equity portfolio.

    Note on COVID Crash: this scenario predates the V2.1 walk-forward
    backtest start (February 2021). It will produce NaN for dynamic
    strategies and is included here for completeness and V1 static analysis.

    Returns
    -------
    dict mapping scenario name to (start_date, end_date).
    """
    return {
        "COVID Crash (Feb–Mar 2020)":      ("2020-02-19", "2020-03-23"),
        "Rate Shock 2022 (Jan–Dec 2022)":   ("2022-01-03", "2022-12-28"),
        "AI Correction 2025 (Feb–Apr 2025)": ("2025-02-19", "2025-04-08"),
    }


# Keep legacy alias for backward compatibility
def define_stress_scenarios() -> Dict[str, Tuple[str, str]]:
    """Alias for define_historical_scenarios()."""
    return define_historical_scenarios()


def compute_stress_metrics(
    strategy_returns: Dict[str, pd.Series],
    scenarios: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """Compute stress metrics for each strategy during each historical scenario.

    For each (scenario, strategy) pair computes:
    - Cumulative Return over the scenario window
    - Annualized Volatility during the scenario
    - Max Drawdown within the scenario window
    - Calmar Ratio (annualized return / |max drawdown|)

    Parameters
    ----------
    strategy_returns:
        Dict mapping strategy name to pd.Series of daily returns
        with a DatetimeIndex.
    scenarios:
        Dict mapping scenario name to (start_date, end_date) strings.

    Returns
    -------
    DataFrame with MultiIndex (Scenario, Strategy) and metric columns.
    """
    results = []

    for scenario_name, (start_date, end_date) in scenarios.items():
        for strategy_name, returns in strategy_returns.items():
            mask = (returns.index >= start_date) & (returns.index <= end_date)
            scenario_returns = returns[mask]

            if len(scenario_returns) == 0:
                results.append({
                    "Scenario": scenario_name,
                    "Strategy": strategy_name,
                    "Cumulative Return": np.nan,
                    "Annualized Vol": np.nan,
                    "Max Drawdown": np.nan,
                    "Calmar Ratio": np.nan,
                    "N Days": 0,
                    "Note": "No data — scenario predates backtest start",
                })
                continue

            cumul_return = float((1 + scenario_returns).prod() - 1)
            ann_vol = float(scenario_returns.std() * np.sqrt(TRADING_DAYS))

            cumul_perf = (1 + scenario_returns).cumprod()
            running_max = cumul_perf.expanding().max()
            drawdown = (cumul_perf / running_max) - 1
            max_dd = float(drawdown.min())

            n_days = len(scenario_returns)
            ann_return = float((1 + cumul_return) ** (TRADING_DAYS / n_days) - 1)
            calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-6 else np.nan

            results.append({
                "Scenario": scenario_name,
                "Strategy": strategy_name,
                "Cumulative Return": cumul_return,
                "Annualized Vol": ann_vol,
                "Max Drawdown": max_dd,
                "Calmar Ratio": calmar,
                "N Days": n_days,
                "Note": "",
            })

    df = pd.DataFrame(results).set_index(["Scenario", "Strategy"])
    return df


def stress_summary_table(stress_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot stress results: scenarios as rows, strategies × metrics as columns."""
    return stress_df.drop(columns=["Note"], errors="ignore").unstack()


def run_stress_analysis(
    strategy_returns: Dict[str, pd.Series],
    scenarios: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """Orchestrate stress analysis and return results DataFrame."""
    return compute_stress_metrics(strategy_returns, scenarios)


def export_stress_results(
    stress_df: pd.DataFrame,
    output_dir: Path,
    version: str = "v2_historical",
) -> None:
    """Export detailed and pivoted stress results to CSV.

    Parameters
    ----------
    stress_df:
        Output of compute_stress_metrics.
    output_dir:
        Directory to write files into.
    version:
        Suffix appended to filenames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stress_df.to_csv(output_dir / f"stress_metrics_{version}.csv")
    stress_summary_table(stress_df).to_csv(output_dir / f"stress_summary_{version}.csv")


def print_stress_summary(stress_df: pd.DataFrame) -> None:
    """Print a clean, readable stress test summary to the terminal."""
    print("\n" + "=" * 72)
    print("STRESS TEST — HISTORICAL SCENARIOS")
    print("=" * 72)

    # Print per-scenario blocks
    scenarios = stress_df.index.get_level_values("Scenario").unique()
    for scenario in scenarios:
        block = stress_df.loc[scenario]
        print(f"\n  {scenario}")
        print(f"  {'Strategy':<22} {'Cumul. Return':>14} {'Ann. Vol':>10} {'Max DD':>10} {'Calmar':>8}")
        print(f"  {'-'*65}")
        for strategy in block.index:
            row = block.loc[strategy]
            if row["N Days"] == 0:
                print(f"  {strategy:<22} {'— no data (predates backtest)':>44}")
            else:
                cr = f"{row['Cumulative Return']:+.1%}"
                vol = f"{row['Annualized Vol']:.1%}"
                dd = f"{row['Max Drawdown']:+.1%}"
                cal = f"{row['Calmar Ratio']:+.2f}" if not np.isnan(row["Calmar Ratio"]) else "n/a"
                print(f"  {strategy:<22} {cr:>14} {vol:>10} {dd:>10} {cal:>8}")

    print("\n" + "=" * 72 + "\n")
