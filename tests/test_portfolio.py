"""
Test suite for lseg-thematic-portfolio-optimization.

Scope: mathematical correctness of optimizers, metrics, and rebalancer logic.
No LSEG API calls — all tests use synthetic data only.
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    rolling_sharpe,
    sharpe_ratio,
)
from src.portfolio import (
    equal_risk_contribution_portfolio,
    equal_weight_portfolio,
    max_sharpe_portfolio,
    min_variance_portfolio,
    min_cvar_portfolio,
    compute_historical_cvar,
    portfolio_volatility,
    risk_contribution,
    weights_to_series,
)
from src.backtest import cumulative_performance, portfolio_returns
from src.rebalancer import walk_forward_rebalance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_simple_cov(vols: list[float], correlation: float = 0.0) -> np.ndarray:
    """Build a covariance matrix from annualized vols and a constant correlation."""
    n = len(vols)
    vols = np.array(vols)
    corr = np.full((n, n), correlation)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    return cov


def make_synthetic_returns(n_assets: int, n_days: int, seed: int = 42) -> pd.DataFrame:
    """Build a clean synthetic daily returns DataFrame (no NaN)."""
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0005, scale=0.015, size=(n_days, n_assets))
    tickers = [f"A{i}" for i in range(n_assets)]
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    return pd.DataFrame(data, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# Test 1 — equal_weight_portfolio
# ---------------------------------------------------------------------------

class TestEqualWeight:

    def test_weights_sum_to_one(self):
        for n in [5, 10, 17, 20]:
            w = equal_weight_portfolio(n)
            assert abs(w.sum() - 1.0) < 1e-10, f"Sum != 1 for n={n}"

    def test_all_weights_equal(self):
        for n in [5, 20]:
            w = equal_weight_portfolio(n)
            assert np.allclose(w, 1.0 / n), f"Weights not equal for n={n}"

    def test_output_shape(self):
        w = equal_weight_portfolio(7)
        assert w.shape == (7,)


# ---------------------------------------------------------------------------
# Test 2 — min_variance_portfolio
# ---------------------------------------------------------------------------

class TestMinVariance:

    def test_sum_to_one(self):
        cov = make_simple_cov([0.2, 0.4, 0.3])
        w = min_variance_portfolio(cov, max_weight=0.5)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_overweights_low_vol_asset(self):
        """With zero correlation, the lowest-vol asset should get highest weight."""
        # Asset 0: vol=10%, Asset 1: vol=40%, Asset 2: vol=30%
        cov = make_simple_cov([0.10, 0.40, 0.30], correlation=0.0)
        w = min_variance_portfolio(cov, max_weight=1.0)
        assert w[0] > w[1], "Low-vol asset should outweigh high-vol asset"
        assert w[0] > w[2], "Low-vol asset should outweigh medium-vol asset"

    def test_respects_max_weight_constraint(self):
        cov = make_simple_cov([0.1, 0.4, 0.4, 0.4])
        for max_w in [0.25, 0.40, 0.50]:
            w = min_variance_portfolio(cov, max_weight=max_w)
            assert w.max() <= max_w + 1e-6, f"Max weight exceeded for max_w={max_w}"

    def test_all_weights_non_negative(self):
        cov = make_simple_cov([0.2, 0.3, 0.25, 0.35])
        w = min_variance_portfolio(cov, max_weight=0.5)
        assert (w >= -1e-8).all(), "Negative weights found (long-only violated)"


# ---------------------------------------------------------------------------
# Test 3 — ERC: risk contributions
# ---------------------------------------------------------------------------

class TestERC:

    def test_risk_contributions_sum_to_portfolio_vol(self):
        """
        Mathematical identity: sum of risk contributions = portfolio volatility.
        This is the core property of the Euler decomposition.
        """
        cov = make_simple_cov([0.20, 0.30, 0.25], correlation=0.3)
        w = np.array([0.4, 0.35, 0.25])
        rc = risk_contribution(w, cov)
        port_vol = portfolio_volatility(w, cov)
        assert abs(rc.sum() - port_vol) < 1e-8, (
            f"RC sum {rc.sum():.8f} != portfolio vol {port_vol:.8f}"
        )

    def test_erc_weights_sum_to_one(self):
        cov = make_simple_cov([0.20, 0.35, 0.28, 0.22], correlation=0.2)
        w = equal_risk_contribution_portfolio(cov, max_weight=0.5)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_erc_weights_non_negative(self):
        cov = make_simple_cov([0.20, 0.35, 0.28, 0.22], correlation=0.2)
        w = equal_risk_contribution_portfolio(cov, max_weight=0.5)
        assert (w >= -1e-8).all()

    def test_erc_differs_from_equal_weight_on_heterogeneous_vols(self):
        """
        With very different vols, ERC should differ meaningfully from EW.
        Low-vol assets should get higher weight in ERC than in EW.
        """
        vols = [0.10, 0.40, 0.10, 0.40]
        cov = make_simple_cov(vols, correlation=0.0)
        w_erc = equal_risk_contribution_portfolio(cov, max_weight=0.5)
        w_ew = equal_weight_portfolio(4)

        # Low-vol assets (index 0 and 2) should be overweighted vs EW
        assert w_erc[0] > w_ew[0] + 0.01, "ERC should overweight low-vol asset vs EW"
        assert w_erc[2] > w_ew[2] + 0.01, "ERC should overweight low-vol asset vs EW"

    def test_erc_risk_contributions_are_balanced(self):
        """
        After ERC optimization, all risk contributions should be approximately equal.
        """
        vols = [0.10, 0.40, 0.25, 0.20]
        cov = make_simple_cov(vols, correlation=0.1)
        w = equal_risk_contribution_portfolio(cov, max_weight=0.5)
        rc = risk_contribution(w, cov)
        # Max deviation from average RC should be small
        avg_rc = rc.mean()
        max_deviation = np.abs(rc - avg_rc).max()
        assert max_deviation < 0.005, (
            f"Risk contributions not balanced: {rc}, max deviation={max_deviation:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 4 — metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_annualized_return_geometric(self):
        """
        A constant daily return of r should compound to (1+r)^252 - 1 annualized.
        """
        daily_r = 0.001
        n = 252
        returns = pd.Series([daily_r] * n)
        expected = (1 + daily_r) ** 252 - 1
        result = annualized_return(returns)
        assert abs(result - expected) < 1e-10

    def test_annualized_volatility(self):
        """Daily vol * sqrt(252) = annualized vol for constant series."""
        returns = pd.Series([0.01, -0.01, 0.02, -0.02] * 63)
        daily_std = returns.std()
        expected = daily_std * np.sqrt(252)
        result = annualized_volatility(returns)
        assert abs(result - expected) < 1e-10

    def test_sharpe_ratio_positive_for_positive_excess_return(self):
        returns = pd.Series([0.001] * 252)
        sr = sharpe_ratio(returns, risk_free_rate=0.0)
        assert sr > 0

    def test_max_drawdown_is_negative(self):
        """Max drawdown should always be <= 0."""
        returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        cumul = cumulative_performance(returns)
        dd = max_drawdown(cumul)
        assert dd <= 0.0

    def test_max_drawdown_flat_series(self):
        """A monotonically increasing series has 0 drawdown."""
        returns = pd.Series([0.001] * 100)
        cumul = cumulative_performance(returns)
        dd = max_drawdown(cumul)
        assert abs(dd) < 1e-10

    def test_portfolio_turnover_full_rotation(self):
        """Rotating from [1,0] to [0,1] has turnover = 2 (sold 1 + bought 1)."""
        old = pd.Series([1.0, 0.0], index=["A", "B"])
        new = pd.Series([0.0, 1.0], index=["A", "B"])
        from src.metrics import portfolio_turnover
        t = portfolio_turnover(old, new)
        assert abs(t - 2.0) < 1e-10

    def test_portfolio_turnover_no_change(self):
        old = pd.Series([0.5, 0.5], index=["A", "B"])
        new = pd.Series([0.5, 0.5], index=["A", "B"])
        from src.metrics import portfolio_turnover
        t = portfolio_turnover(old, new)
        assert abs(t) < 1e-10


# ---------------------------------------------------------------------------
# Test 5 — walk_forward_rebalance (structural)
# ---------------------------------------------------------------------------

class TestWalkForwardRebalance:

    def test_returns_three_outputs(self):
        returns = make_synthetic_returns(n_assets=5, n_days=500)
        result = walk_forward_rebalance(
            returns=returns,
            optimizer_func=equal_weight_portfolio,
            optimizer_name="equal_weight",
            lookback_window_days=120,
            rebalance_frequency="ME",
            transaction_cost_bps=10,
            max_weight=0.4,
        )
        assert len(result) == 3, "walk_forward_rebalance must return exactly 3 values"

    def test_portfolio_returns_not_empty(self):
        returns = make_synthetic_returns(n_assets=5, n_days=500)
        port_returns, weights_history, meta = walk_forward_rebalance(
            returns=returns,
            optimizer_func=equal_weight_portfolio,
            optimizer_name="equal_weight",
            lookback_window_days=120,
            rebalance_frequency="ME",
            transaction_cost_bps=10,
            max_weight=0.4,
        )
        assert not port_returns.empty, "Portfolio returns should not be empty"

    def test_weights_sum_to_one_at_each_rebalance(self):
        returns = make_synthetic_returns(n_assets=5, n_days=500)
        _, weights_history, _ = walk_forward_rebalance(
            returns=returns,
            optimizer_func=min_variance_portfolio,
            optimizer_name="min_variance",
            lookback_window_days=120,
            rebalance_frequency="ME",
            transaction_cost_bps=10,
            max_weight=0.4,
        )
        assert not weights_history.empty
        row_sums = weights_history.sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-6, (
            "Weights should sum to 1 at each rebalance"
        )

    def test_meta_has_expected_columns(self):
        returns = make_synthetic_returns(n_assets=5, n_days=500)
        _, _, meta = walk_forward_rebalance(
            returns=returns,
            optimizer_func=equal_weight_portfolio,
            optimizer_name="equal_weight",
            lookback_window_days=120,
            rebalance_frequency="ME",
            transaction_cost_bps=10,
            max_weight=0.4,
        )
        expected_cols = {"rebalance_date", "optimizer_name", "n_valid_assets", "used_equal_weight_fallback"}
        assert expected_cols.issubset(set(meta.columns)), (
            f"Missing meta columns: {expected_cols - set(meta.columns)}"
        )

    def test_insufficient_data_returns_empty(self):
        """If returns is shorter than lookback, should return empty gracefully."""
        returns = make_synthetic_returns(n_assets=5, n_days=50)
        port_returns, weights_history, meta = walk_forward_rebalance(
            returns=returns,
            optimizer_func=equal_weight_portfolio,
            optimizer_name="equal_weight",
            lookback_window_days=252,
            rebalance_frequency="ME",
            transaction_cost_bps=10,
            max_weight=0.4,
        )
        assert port_returns.empty, "Should return empty series if insufficient data"

    def test_no_fallback_on_clean_data(self):
        """With perfectly clean data (no NaN), fallback should never be triggered."""
        returns = make_synthetic_returns(n_assets=5, n_days=600)
        _, _, meta = walk_forward_rebalance(
            returns=returns,
            optimizer_func=min_variance_portfolio,
            optimizer_name="min_variance",
            lookback_window_days=120,
            rebalance_frequency="ME",
            transaction_cost_bps=10,
            max_weight=0.4,
        )
        assert not meta.empty
        assert meta["used_equal_weight_fallback"].sum() == 0, (
            "No fallback expected on clean synthetic data"
        )


# ---------------------------------------------------------------------------
# Test 7 — CVaR functions
# ---------------------------------------------------------------------------

class TestCVaR:

    def test_compute_historical_cvar_is_less_extreme_than_max_loss(self):
        """CVaR should be between the percentile loss and the minimum loss."""
        returns = np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
        cvar = compute_historical_cvar(returns, confidence_level=0.90)
        
        # At 90% confidence, we look at the worst 10% (1 observation)
        # CVaR should be <= max loss and >= the worst observation in the tail
        assert cvar <= returns.min() + 1e-10
        assert cvar >= returns.min() - 1e-10

    def test_compute_historical_cvar_95_vs_90(self):
        """
        At higher confidence (95% vs 90%), we look at deeper into the tail,
        so CVaR should be more negative (more extreme).
        E.g.: 90% confidence = worst 10%, 95% confidence = worst 5% (deeper tail).
        """
        np.random.seed(42)
        returns = np.random.normal(-0.001, 0.02, size=1000)
        cvar_90 = compute_historical_cvar(returns, confidence_level=0.90)
        cvar_95 = compute_historical_cvar(returns, confidence_level=0.95)
        
        # At 95% confidence, we go deeper into tail, so CVaR is more negative
        assert cvar_95 <= cvar_90, f"CVaR(95%)={cvar_95} should be <= CVaR(90%)={cvar_90}"

    def test_min_cvar_portfolio_weights_sum_to_one(self):
        returns_df = make_synthetic_returns(n_assets=4, n_days=252, seed=42)
        w = min_cvar_portfolio(returns_df, max_weight=0.5, confidence_level=0.95)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_min_cvar_portfolio_weights_non_negative(self):
        returns_df = make_synthetic_returns(n_assets=5, n_days=252, seed=42)
        w = min_cvar_portfolio(returns_df, max_weight=0.4, confidence_level=0.95)
        assert (w >= -1e-8).all(), "Long-only constraint violated"

    def test_min_cvar_portfolio_respects_max_weight(self):
        returns_df = make_synthetic_returns(n_assets=4, n_days=252, seed=42)
        for max_w in [0.25, 0.33, 0.5, 1.0]:
            w = min_cvar_portfolio(returns_df, max_weight=max_w, confidence_level=0.95)
            assert w.max() <= max_w + 1e-6, f"Max weight {w.max():.6f} exceeds {max_w}"

    def test_min_cvar_portfolio_capped_vs_uncapped(self):
        """
        The uncapped (max_weight=1.0) version may show more concentration,
        while the capped (max_weight=0.2) version should be more diversified.
        """
        np.random.seed(123)
        returns_df = make_synthetic_returns(n_assets=6, n_days=252, seed=123)
        
        w_capped = min_cvar_portfolio(returns_df, max_weight=0.20, confidence_level=0.95)
        w_uncapped = min_cvar_portfolio(returns_df, max_weight=1.0, confidence_level=0.95)
        
        # Uncapped should have higher Herfindahl (concentration)
        hh_capped = (w_capped ** 2).sum()
        hh_uncapped = (w_uncapped ** 2).sum()
        
        # Generally true, but not guaranteed; concentration difference should be non-trivial
        assert hh_uncapped >= hh_capped - 1e-6, (
            f"Uncapped Herfindahl {hh_uncapped:.6f} should >= capped {hh_capped:.6f}"
        )

    def test_min_cvar_portfolio_with_extreme_confidence_levels(self):
        """Test edge cases: very high and very low confidence levels."""
        returns_df = make_synthetic_returns(n_assets=3, n_days=200, seed=42)
        
        # Very high confidence (only worst 0.5%)
        w_high = min_cvar_portfolio(returns_df, max_weight=0.5, confidence_level=0.995)
        assert abs(w_high.sum() - 1.0) < 1e-6
        
        # Lower confidence (worst 20%)
        w_low = min_cvar_portfolio(returns_df, max_weight=0.5, confidence_level=0.80)
        assert abs(w_low.sum() - 1.0) < 1e-6

    def test_min_cvar_portfolio_walk_forward_integration(self):
        """Test that min_cvar works in walk-forward rebalancing."""
        returns = make_synthetic_returns(n_assets=4, n_days=500, seed=42)
        
        portfolio_returns, weights_hist, meta = walk_forward_rebalance(
            returns=returns,
            optimizer_func=min_cvar_portfolio,
            optimizer_name="min_cvar",
            lookback_window_days=120,
            rebalance_frequency="ME",
            transaction_cost_bps=10,
            max_weight=0.30,
        )
        
        assert len(portfolio_returns) > 0, "No portfolio returns generated"
        assert not weights_hist.empty, "Weights history is empty"
        assert not meta.empty, "Metadata is empty"
        # Check no NaN values in returns
        assert not np.any(np.isnan(portfolio_returns.values)), "Portfolio returns contain NaN"
        