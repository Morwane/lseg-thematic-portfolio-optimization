"""Main orchestration script for portfolio optimization V2.1."""

from pathlib import Path

import pandas as pd

from src.backtest import cumulative_performance
from src.data_fetcher import MarketDataRequest, fetch_prices_lseg
from src.factor_analysis import (
    compute_factor_exposures,
    export_factor_exposures,
    print_factor_summary,
)
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
    equal_weight_portfolio,
    equal_risk_contribution_portfolio,
    max_sharpe_portfolio,
    min_variance_portfolio,
    min_cvar_portfolio,
    weights_to_series,
)
from src.preprocessing import clean_prices, compute_returns
from src.rebalancer import walk_forward_rebalance
from src.stress import (
    define_historical_scenarios,
    run_stress_analysis,
    export_stress_results,
    print_stress_summary,
)
from src.utils import load_config
from src.visualization import (
    plot_cumulative_performance,
    plot_drawdown_series,
    plot_efficient_frontier,
    plot_rolling_sharpe,
    plot_rolling_volatility,
    plot_weights,
    plot_stress_comparison,
    plot_stress_heatmap,
    plot_erc_vs_ew_capital_vs_risk,
    plot_erc_weight_stability,
)


def main() -> None:
    """Orchestrate V2.1 dynamic portfolio optimization workflow with monthly rebalancing."""
    config = load_config("config/settings.yaml")

    tickers = config["tickers"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    risk_free_rate = float(config["risk_free_rate"])
    max_weight = float(config["max_weight"])
    rebalance_frequency = config["rebalance_frequency"]
    lookback_window_days = int(config["lookback_window_days"])
    transaction_cost_bps = float(config["transaction_cost_bps"])
    rolling_window_days = int(config["rolling_window_days"])
    covariance_method = config.get("covariance_method", "sample")

    # --- Data fetching and preprocessing ---
    request = MarketDataRequest(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    prices = fetch_prices_lseg(request)

    prices_clean = clean_prices(prices)

    if isinstance(prices_clean.columns, pd.MultiIndex):
        prices_clean = prices_clean.copy()
        prices_clean.columns = [str(col[0]) for col in prices_clean.columns]

    returns = compute_returns(prices_clean)

    # --- Walk-forward rebalancing ---
    print(f"\nRunning walk-forward strategies (covariance_method='{covariance_method}')...")

    ew_returns, ew_weights_history, ew_meta = walk_forward_rebalance(
        returns=returns,
        optimizer_func=equal_weight_portfolio,
        optimizer_name="equal_weight",
        lookback_window_days=lookback_window_days,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=transaction_cost_bps,
        max_weight=max_weight,
        risk_free_rate=risk_free_rate,
        covariance_method=covariance_method,
    )

    mv_returns, mv_weights_history, mv_meta = walk_forward_rebalance(
        returns=returns,
        optimizer_func=min_variance_portfolio,
        optimizer_name="min_variance",
        lookback_window_days=lookback_window_days,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=transaction_cost_bps,
        max_weight=max_weight,
        risk_free_rate=risk_free_rate,
        covariance_method=covariance_method,
    )

    ms_returns, ms_weights_history, ms_meta = walk_forward_rebalance(
        returns=returns,
        optimizer_func=max_sharpe_portfolio,
        optimizer_name="max_sharpe",
        lookback_window_days=lookback_window_days,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=transaction_cost_bps,
        max_weight=max_weight,
        risk_free_rate=risk_free_rate,
        covariance_method=covariance_method,
    )

    erc_returns, erc_weights_history, erc_meta = walk_forward_rebalance(
        returns=returns,
        optimizer_func=equal_risk_contribution_portfolio,
        optimizer_name="erc",
        lookback_window_days=lookback_window_days,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=transaction_cost_bps,
        max_weight=max_weight,
        risk_free_rate=risk_free_rate,
        covariance_method=covariance_method,
    )

    print("  Computing Min CVaR portfolio (capped 20%)...")
    cvar_returns, cvar_weights_history, cvar_meta = walk_forward_rebalance(
        returns=returns,
        optimizer_func=min_cvar_portfolio,
        optimizer_name="min_cvar",
        lookback_window_days=lookback_window_days,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=transaction_cost_bps,
        max_weight=max_weight,
        risk_free_rate=risk_free_rate,
        covariance_method=covariance_method,
    )

    # --- Cumulative performance and drawdowns ---
    ew_cumul = cumulative_performance(ew_returns.copy())
    mv_cumul = cumulative_performance(mv_returns.copy())
    ms_cumul = cumulative_performance(ms_returns.copy())
    erc_cumul = cumulative_performance(erc_returns.copy())
    cvar_cumul = cumulative_performance(cvar_returns.copy())

    ew_drawdown = drawdown_series(ew_cumul)
    mv_drawdown = drawdown_series(mv_cumul)
    ms_drawdown = drawdown_series(ms_cumul)
    erc_drawdown = drawdown_series(erc_cumul)
    cvar_drawdown = drawdown_series(cvar_cumul)

    drawdown_df = pd.DataFrame({
        "Equal Weight": ew_drawdown,
        "Min Variance": mv_drawdown,
        "Max Sharpe": ms_drawdown,
        "ERC": erc_drawdown,
        "Min CVaR": cvar_drawdown,
    })

    cumul_df = pd.DataFrame({
        "Equal Weight": ew_cumul,
        "Min Variance": mv_cumul,
        "Max Sharpe": ms_cumul,
        "ERC": erc_cumul,
        "Min CVaR": cvar_cumul,
    })

    # --- Rolling metrics ---
    rolling_sharpe_df = pd.DataFrame({
        "Equal Weight": rolling_sharpe(ew_returns, window=rolling_window_days, risk_free_rate=risk_free_rate),
        "Min Variance": rolling_sharpe(mv_returns, window=rolling_window_days, risk_free_rate=risk_free_rate),
        "Max Sharpe": rolling_sharpe(ms_returns, window=rolling_window_days, risk_free_rate=risk_free_rate),
        "ERC": rolling_sharpe(erc_returns, window=rolling_window_days, risk_free_rate=risk_free_rate),
        "Min CVaR": rolling_sharpe(cvar_returns, window=rolling_window_days, risk_free_rate=risk_free_rate),
    })

    rolling_vol_df = pd.DataFrame({
        "Equal Weight": rolling_volatility(ew_returns, window=rolling_window_days),
        "Min Variance": rolling_volatility(mv_returns, window=rolling_window_days),
        "Max Sharpe": rolling_volatility(ms_returns, window=rolling_window_days),
        "ERC": rolling_volatility(erc_returns, window=rolling_window_days),
        "Min CVaR": rolling_volatility(cvar_returns, window=rolling_window_days),
    })

    # --- Summary metrics ---
    def _calmar(ret_series, cumul_series):
        ann_ret = annualized_return(ret_series)
        dd = max_drawdown(cumul_series)
        return ann_ret / abs(dd) if abs(dd) > 1e-8 else float("nan")

    def _effective_n(weights_history: pd.DataFrame) -> float:
        """Average effective number of assets (inverse HHI) across rebalance windows."""
        if weights_history.empty:
            return float("nan")
        def eff_n_row(row):
            w = row[row > 0.001]
            return 1.0 / (w ** 2).sum() if len(w) > 0 else float("nan")
        return weights_history.apply(eff_n_row, axis=1).mean()

    summary_data = {
        "Portfolio": ["Equal Weight", "Min Variance", "Max Sharpe", "ERC", "Min CVaR"],
        "Annualized Return": [
            annualized_return(ew_returns),
            annualized_return(mv_returns),
            annualized_return(ms_returns),
            annualized_return(erc_returns),
            annualized_return(cvar_returns),
        ],
        "Annualized Volatility": [
            annualized_volatility(ew_returns),
            annualized_volatility(mv_returns),
            annualized_volatility(ms_returns),
            annualized_volatility(erc_returns),
            annualized_volatility(cvar_returns),
        ],
        "Sharpe Ratio": [
            sharpe_ratio(ew_returns, risk_free_rate),
            sharpe_ratio(mv_returns, risk_free_rate),
            sharpe_ratio(ms_returns, risk_free_rate),
            sharpe_ratio(erc_returns, risk_free_rate),
            sharpe_ratio(cvar_returns, risk_free_rate),
        ],
        "Max Drawdown": [
            max_drawdown(ew_cumul),
            max_drawdown(mv_cumul),
            max_drawdown(ms_cumul),
            max_drawdown(erc_cumul),
            max_drawdown(cvar_cumul),
        ],
        "Calmar Ratio": [
            _calmar(ew_returns, ew_cumul),
            _calmar(mv_returns, mv_cumul),
            _calmar(ms_returns, ms_cumul),
            _calmar(erc_returns, erc_cumul),
            _calmar(cvar_returns, cvar_cumul),
        ],
        "Avg Effective N Assets": [
            _effective_n(ew_weights_history),
            _effective_n(mv_weights_history),
            _effective_n(ms_weights_history),
            _effective_n(erc_weights_history),
            _effective_n(cvar_weights_history),
        ],
        "Worst Drawdown Date": [
            ew_drawdown.idxmin() if not ew_drawdown.empty else pd.NaT,
            mv_drawdown.idxmin() if not mv_drawdown.empty else pd.NaT,
            ms_drawdown.idxmin() if not ms_drawdown.empty else pd.NaT,
            erc_drawdown.idxmin() if not erc_drawdown.empty else pd.NaT,
            cvar_drawdown.idxmin() if not cvar_drawdown.empty else pd.NaT,
        ],
    }

    summary_df = pd.DataFrame(summary_data).set_index("Portfolio")

    # --- Final weights for plots (last rebalance) ---
    def last_weights(history: pd.DataFrame, fallback_tickers: list) -> pd.Series:
        if not history.empty:
            return history.iloc[-1].reindex(fallback_tickers).fillna(0.0)
        return pd.Series(0.0, index=fallback_tickers, dtype=float)

    ew_final = last_weights(ew_weights_history, tickers)
    mv_final = last_weights(mv_weights_history, tickers)
    ms_final = last_weights(ms_weights_history, tickers)
    erc_final = last_weights(erc_weights_history, tickers)
    cvar_final = last_weights(cvar_weights_history, tickers)

    # --- Efficient frontier (full dataset) ---
    mean_returns_full = returns.mean().values * TRADING_DAYS
    cov_matrix_full = returns.cov().values * TRADING_DAYS

    # --- Valid asset summary (from EW meta — same filtering for all) ---
    if not ew_meta.empty:
        valid_assets_summary = ew_meta[
            ["rebalance_date", "n_valid_assets", "used_equal_weight_fallback"]
        ].copy()
        valid_assets_summary.columns = ["Rebalance Date", "N Valid Assets", "Used EW Fallback"]

    # --- Export outputs ---
    Path("output/reports").mkdir(parents=True, exist_ok=True)
    Path("output/charts").mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "Equal Weight": ew_returns,
        "Min Variance": mv_returns,
        "Max Sharpe": ms_returns,
        "ERC": erc_returns,
        "Min CVaR": cvar_returns,
    }).to_csv("output/reports/portfolio_returns_v2.csv")

    ew_weights_history.to_csv("output/reports/ew_weights_history_v2.csv")
    mv_weights_history.to_csv("output/reports/mv_weights_history_v2.csv")
    ms_weights_history.to_csv("output/reports/ms_weights_history_v2.csv")
    erc_weights_history.to_csv("output/reports/erc_weights_history_v2.csv")
    cvar_weights_history.to_csv("output/reports/cvar_weights_history_v2.csv")

    ew_meta.to_csv("output/reports/meta_ew_v2.csv", index=False)
    mv_meta.to_csv("output/reports/meta_mv_v2.csv", index=False)
    ms_meta.to_csv("output/reports/meta_ms_v2.csv", index=False)
    erc_meta.to_csv("output/reports/meta_erc_v2.csv", index=False)
    cvar_meta.to_csv("output/reports/meta_cvar_v2.csv", index=False)

    if not ew_meta.empty:
        valid_assets_summary.to_csv("output/reports/valid_assets_per_window_v2.csv", index=False)

    summary_df.to_csv("output/reports/portfolio_summary_v2.csv")
    cumul_df.to_csv("output/reports/cumulative_performance_v2.csv")
    drawdown_df.to_csv("output/reports/drawdowns_v2.csv")
    rolling_sharpe_df.to_csv("output/reports/rolling_sharpe_v2.csv")
    rolling_vol_df.to_csv("output/reports/rolling_volatility_v2.csv")

    # --- Plots ---
    plot_cumulative_performance(cumul_df, output_path="output/charts/cumulative_performance_v2.png")
    plot_drawdown_series(drawdown_df, output_path="output/charts/drawdowns_v2.png")
    plot_weights(
        {"Equal Weight": ew_final, "Min Variance": mv_final,
         "Max Sharpe": ms_final, "ERC": erc_final},
        output_path="output/charts/portfolio_weights_v2.png",
    )
    plot_efficient_frontier(
        mean_returns_full, cov_matrix_full,
        {"Equal Weight": ew_final.values, "Min Variance": mv_final.values,
         "Max Sharpe": ms_final.values, "ERC": erc_final.values},
        risk_free_rate,
        output_path="output/charts/efficient_frontier_v2.png",
    )
    plot_rolling_sharpe(rolling_sharpe_df, output_path="output/charts/rolling_sharpe_v2.png")
    plot_rolling_volatility(rolling_vol_df, output_path="output/charts/rolling_volatility_v2.png")

    # --- ERC flagship charts ---
    plot_erc_vs_ew_capital_vs_risk(
        erc_weights=erc_final,
        ew_weights=ew_final,
        cov_matrix=cov_matrix_full,
        output_path="output/charts/erc_vs_ew_capital_risk.png",
    )
    plot_erc_weight_stability(
        erc_weights_history=erc_weights_history,
        output_path="output/charts/erc_weight_stability.png",
    )

    # --- Terminal summary ---
    print("\n=== V2.1 Dynamic Portfolio Summary ===\n")
    print(summary_df.round(4))

    print("\n=== Optimization Meta — EW (representative) ===")
    if not ew_meta.empty:
        fallback_count = ew_meta["used_equal_weight_fallback"].sum()
        total = len(ew_meta)
        print(f"  Total rebalance windows : {total}")
        print(f"  Windows with EW fallback: {fallback_count} ({100*fallback_count/total:.1f}%)")
        print(f"  Windows with real optim : {total - fallback_count} ({100*(total-fallback_count)/total:.1f}%)")
        for n, count in ew_meta["n_valid_assets"].value_counts().sort_index().items():
            print(f"    {n} assets: {count} windows")

    # --- Stress testing (historical scenarios only) ---
    print("\n=== Stress Testing ===")
    strategy_returns_dict = {
        "Equal Weight": ew_returns,
        "Min Variance": mv_returns,
        "Max Sharpe": ms_returns,
        "ERC": erc_returns,
        "Min CVaR": cvar_returns,
    }
    historical_scenarios = define_historical_scenarios()
    stress_df = run_stress_analysis(strategy_returns_dict, historical_scenarios)
    export_stress_results(stress_df, Path("output/reports"), version="v2_historical")
    plot_stress_comparison(stress_df, output_path="output/charts/stress_comparison_v2.png")
    plot_stress_heatmap(
        stress_df, metric="Max Drawdown",
        output_path="output/charts/stress_heatmap_v2_maxdd.png",
    )
    print_stress_summary(stress_df)

    # --- Factor exposure analysis ---
    print("\n=== Factor Exposure Analysis ===")
    if not ew_meta.empty:
        rebalance_dates = pd.to_datetime(ew_meta["rebalance_date"]).unique()
    else:
        rebalance_dates = ew_weights_history.index.unique()

    strategies_dict = {
        "Equal Weight": (ew_returns, ew_weights_history, ew_meta),
        "Min Variance": (mv_returns, mv_weights_history, mv_meta),
        "Max Sharpe": (ms_returns, ms_weights_history, ms_meta),
        "ERC": (erc_returns, erc_weights_history, erc_meta),
        "Min CVaR": (cvar_returns, cvar_weights_history, cvar_meta),
    }

    factor_exposures = compute_factor_exposures(
        prices=prices_clean,
        returns=returns,
        strategies_dict=strategies_dict,
        rebalance_dates=rebalance_dates,
        lookback_window_days=lookback_window_days,
    )
    export_factor_exposures(factor_exposures, output_path="output/reports/factor_exposure_v2.csv")
    print_factor_summary(factor_exposures)

    # --- Black-Litterman (static, on full-period covariance) ---
    # Note: this is a point-in-time analytical computation, NOT a backtest.
    # It shows what BL weights look like given the three analyst views and
    # the full-sample covariance. It is comparable to V1 (static), not V2.1.
    print("\n=== Black-Litterman Portfolio (Static — analytical, not backtested) ===")
    from src.covariance import build_covariance_matrix
    from src.black_litterman import (
        run_black_litterman,
        get_ai_tech_views,
        print_bl_summary,
        export_bl_results,
    )

    bl_tickers = [t for t in tickers if t in returns.columns]
    bl_returns_clean = returns[bl_tickers].dropna()
    bl_cov = build_covariance_matrix(bl_returns_clean, method=covariance_method)
    bl_views = get_ai_tech_views("medium")

    bl_result = run_black_litterman(
        tickers=bl_tickers,
        cov_matrix=bl_cov,
        views=bl_views,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        risk_aversion=2.5,
        tau=0.05,
    )

    export_bl_results(bl_result, output_dir="output/reports")
    print_bl_summary(bl_result)

    # --- Black-Litterman walk-forward (dynamic momentum views) ---
    print("\n=== Black-Litterman Walk-Forward (Dynamic Momentum Views) ===")

    bl_wf_returns, bl_wf_weights_history, bl_wf_meta = walk_forward_rebalance(
        returns=returns,
        optimizer_func=None,           # dispatch handled internally via optimizer_name
        optimizer_name="black_litterman",
        lookback_window_days=lookback_window_days,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=transaction_cost_bps,
        max_weight=max_weight,
        risk_free_rate=risk_free_rate,
        covariance_method=covariance_method,
        bl_risk_aversion=2.5,
        bl_tau=0.05,
    )

    if not bl_wf_returns.empty:
        bl_wf_cumul = cumulative_performance(bl_wf_returns.copy())
        bl_wf_drawdown = drawdown_series(bl_wf_cumul)

        bl_wf_ann_ret = annualized_return(bl_wf_returns)
        bl_wf_ann_vol = annualized_volatility(bl_wf_returns)
        bl_wf_sharpe = sharpe_ratio(bl_wf_returns, risk_free_rate)
        bl_wf_maxdd = max_drawdown(bl_wf_cumul)
        bl_wf_calmar = bl_wf_ann_ret / abs(bl_wf_maxdd) if abs(bl_wf_maxdd) > 1e-8 else float("nan")

        print(f"  Annualized Return : {bl_wf_ann_ret:.2%}")
        print(f"  Annualized Vol    : {bl_wf_ann_vol:.2%}")
        print(f"  Sharpe Ratio      : {bl_wf_sharpe:.2f}")
        print(f"  Max Drawdown      : {bl_wf_maxdd:.2%}")
        print(f"  Calmar Ratio      : {bl_wf_calmar:.2f}")

        bl_wf_weights_history.to_csv("output/reports/bl_wf_weights_history_v2.csv")
        bl_wf_meta.to_csv("output/reports/bl_wf_meta_v2.csv", index=False)

        # --- BL cumulative performance plot (with all strategies) ---
        cumul_df_with_bl = cumul_df.copy()
        cumul_df_with_bl["BL Walk-Forward"] = bl_wf_cumul
        plot_cumulative_performance(
            cumul_df_with_bl,
            output_path="output/charts/cumulative_performance_v2_with_bl.png",
        )

        # --- BL drawdown chart ---
        drawdown_df_with_bl = drawdown_df.copy()
        drawdown_df_with_bl["BL Walk-Forward"] = bl_wf_drawdown
        plot_drawdown_series(
            drawdown_df_with_bl,
            output_path="output/charts/drawdowns_v2_with_bl.png",
        )

        # --- BL rolling volatility ---
        bl_wf_rolling_vol = rolling_volatility(bl_wf_returns, window=rolling_window_days)
        rolling_vol_df_with_bl = rolling_vol_df.copy()
        rolling_vol_df_with_bl["BL Walk-Forward"] = bl_wf_rolling_vol
        plot_rolling_volatility(
            rolling_vol_df_with_bl,
            output_path="output/charts/rolling_volatility_v2_with_bl.png",
        )

        # --- BL rolling Sharpe ---
        bl_wf_rolling_sharpe = rolling_sharpe(bl_wf_returns, window=rolling_window_days, risk_free_rate=risk_free_rate)
        rolling_sharpe_df_with_bl = rolling_sharpe_df.copy()
        rolling_sharpe_df_with_bl["BL Walk-Forward"] = bl_wf_rolling_sharpe
        plot_rolling_sharpe(
            rolling_sharpe_df_with_bl,
            output_path="output/charts/rolling_sharpe_v2_with_bl.png",
        )

        # --- BL stress testing ---
        print("\n=== BL Walk-Forward Stress Testing ===")
        strategy_returns_dict_with_bl = {**strategy_returns_dict}
        strategy_returns_dict_with_bl["BL Walk-Forward"] = bl_wf_returns
        stress_df_with_bl = run_stress_analysis(strategy_returns_dict_with_bl, historical_scenarios)
        export_stress_results(stress_df_with_bl, Path("output/reports"), version="v2_historical_with_bl")
        plot_stress_comparison(stress_df_with_bl, output_path="output/charts/stress_comparison_v2_with_bl.png")
        print_stress_summary(stress_df_with_bl)

        # --- BL factor exposure analysis ---
        print("\n=== BL Walk-Forward Factor Exposures ===")
        strategies_dict_with_bl = {**strategies_dict}
        strategies_dict_with_bl["BL Walk-Forward"] = (bl_wf_returns, bl_wf_weights_history, bl_wf_meta)
        factor_exposures_with_bl = compute_factor_exposures(
            prices=prices_clean,
            returns=returns,
            strategies_dict=strategies_dict_with_bl,
            rebalance_dates=rebalance_dates,
            lookback_window_days=lookback_window_days,
        )
        export_factor_exposures(factor_exposures_with_bl, output_path="output/reports/factor_exposure_v2_with_bl.csv")
        print_factor_summary(factor_exposures_with_bl)

        # --- BL summary row for comparison ---
        print("\n=== BL Walk-Forward vs Core Strategies ===")
        bl_summary = pd.DataFrame({
            "Portfolio": ["BL Walk-Forward"],
            "Annualized Return": [bl_wf_ann_ret],
            "Annualized Volatility": [bl_wf_ann_vol],
            "Sharpe Ratio": [bl_wf_sharpe],
            "Max Drawdown": [bl_wf_maxdd],
            "Calmar Ratio": [bl_wf_calmar],
        }).set_index("Portfolio")
        combined_summary = pd.concat([summary_df[["Annualized Return", "Annualized Volatility",
                                                    "Sharpe Ratio", "Max Drawdown", "Calmar Ratio"]],
                                       bl_summary])
        print(combined_summary.round(4))
        combined_summary.to_csv("output/reports/portfolio_summary_v2_with_bl.csv")

    else:
        print("  BL walk-forward produced no returns (insufficient data).")


if __name__ == "__main__":
    main()
