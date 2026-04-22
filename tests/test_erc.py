"""
Focused test suite for Equal Risk Contribution (ERC) portfolio optimization.

Tests cover:
  - Mathematical correctness of the risk decomposition
  - Economic properties of the ERC solution
  - Robustness under constraints and edge cases
  - Walk-forward stability

All tests use synthetic data only — no LSEG credentials required.
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    equal_risk_contribution_portfolio,
    equal_weight_portfolio,
    portfolio_volatility,
    risk_contribution,
)
from src.rebalancer import walk_forward_rebalance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cov_from_vols(vols: list[float], correlation: float = 0.0) -> np.ndarray:
    """Build an n×n covariance matrix from annualised vols and constant correlation."""
    n = len(vols)
    v = np.array(vols)
    corr = np.full((n, n), correlation)
    np.fill_diagonal(corr, 1.0)
    return np.outer(v, v) * corr


def synthetic_returns(
    n_assets: int,
    n_days: int,
    vols_ann: list[float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic daily returns. Optionally pass per-asset annualised volatilities."""
    rng = np.random.default_rng(seed)
    if vols_ann is not None:
        vols_d = np.array(vols_ann) / np.sqrt(252)
        data = rng.normal(0.0003, 1.0, (n_days, n_assets)) * vols_d
    else:
        data = rng.normal(0.0003, 0.015, (n_days, n_assets))
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    tickers = [f"T{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# 1 — Euler decomposition identity
# ---------------------------------------------------------------------------

class TestRiskDecompositionIdentity:
    """The Euler identity must hold for any weight vector: sum(RC_i) == portfolio_vol."""

    def test_identity_holds_for_equal_weight(self):
        cov = cov_from_vols([0.20, 0.30, 0.25], correlation=0.3)
        w = equal_weight_portfolio(3)
        rc = risk_contribution(w, cov)
        vol = portfolio_volatility(w, cov)
        assert abs(rc.sum() - vol) < 1e-9

    def test_identity_holds_for_erc_solution(self):
        cov = cov_from_vols([0.15, 0.45, 0.30, 0.20], correlation=0.2)
        w = equal_risk_contribution_portfolio(cov, max_weight=0.5)
        rc = risk_contribution(w, cov)
        vol = portfolio_volatility(w, cov)
        assert abs(rc.sum() - vol) < 1e-9

    def test_identity_holds_for_random_weights(self):
        rng = np.random.default_rng(0)
        cov = cov_from_vols([0.20, 0.35, 0.28, 0.22, 0.40], correlation=0.15)
        for _ in range(10):
            w_raw = rng.dirichlet(np.ones(5))
            rc = risk_contribution(w_raw, cov)
            vol = portfolio_volatility(w_raw, cov)
            assert abs(rc.sum() - vol) < 1e-9

    def test_identity_zero_vol_asset(self):
        """An asset with zero variance and zero covariances has zero risk contribution."""
        cov = np.array([[0.04, 0.0], [0.0, 0.0]])
        w = np.array([0.5, 0.5])
        rc = risk_contribution(w, cov)
        assert rc[1] < 1e-12


# ---------------------------------------------------------------------------
# 2 — ERC solution properties
# ---------------------------------------------------------------------------

class TestERCSolutionProperties:

    def test_weights_sum_to_one(self):
        cov = cov_from_vols([0.20, 0.35, 0.28, 0.22], correlation=0.2)
        w = equal_risk_contribution_portfolio(cov, max_weight=0.5)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self):
        cov = cov_from_vols([0.20, 0.35, 0.28, 0.22], correlation=0.2)
        w = equal_risk_contribution_portfolio(cov, max_weight=0.5)
        assert (w >= -1e-8).all()

    def test_max_weight_constraint_respected(self):
        cov = cov_from_vols([0.10, 0.50, 0.10, 0.50], correlation=0.0)
        for cap in [0.25, 0.40, 0.50, 1.0]:
            w = equal_risk_contribution_portfolio(cov, max_weight=cap)
            assert w.max() <= cap + 1e-6, f"Cap {cap} violated: max weight = {w.max():.6f}"

    def test_risk_contributions_are_balanced_homogeneous(self):
        """With identical assets, ERC must give equal weights (= equal weight)."""
        cov = cov_from_vols([0.25, 0.25, 0.25, 0.25], correlation=0.3)
        w = equal_risk_contribution_portfolio(cov, max_weight=0.5)
        assert np.allclose(w, 0.25, atol=1e-5), f"Identical assets: expected equal weights, got {w}"

    def test_risk_contributions_balanced_heterogeneous(self):
        """After ERC optimisation, max absolute deviation of RC from mean must be small."""
        vols = [0.10, 0.45, 0.25, 0.20, 0.35]
        cov = cov_from_vols(vols, correlation=0.15)
        w = equal_risk_contribution_portfolio(cov, max_weight=0.6)
        rc = risk_contribution(w, cov)
        rc_frac = rc / rc.sum()  # normalise to fractions
        avg = rc_frac.mean()
        max_dev = np.abs(rc_frac - avg).max()
        # Tolerance of 0.5 pp in risk fraction is tight; SLSQP should beat this
        assert max_dev < 0.005, (
            f"Risk contributions not balanced: fractions={rc_frac.round(5)}, "
            f"max deviation={max_dev:.6f}"
        )


# ---------------------------------------------------------------------------
# 3 — ERC vs Equal Weight economic properties
# ---------------------------------------------------------------------------

class TestERCvsEqualWeight:
    """Core thesis: ERC reweights capital so that risk is spread evenly."""

    def test_high_vol_asset_gets_lower_erc_weight_than_ew(self):
        """A 4× more volatile asset should receive substantially less ERC weight."""
        # Asset 0: vol=10%, Asset 1: vol=40% — 4× higher
        cov = cov_from_vols([0.10, 0.40], correlation=0.0)
        w_erc = equal_risk_contribution_portfolio(cov, max_weight=1.0)
        w_ew = equal_weight_portfolio(2)
        assert w_erc[0] > w_ew[0] + 0.05, (
            f"Low-vol asset should be notably overweighted in ERC: "
            f"ERC={w_erc[0]:.3f} vs EW={w_ew[0]:.3f}"
        )
        assert w_erc[1] < w_ew[1] - 0.05, (
            f"High-vol asset should be notably underweighted in ERC: "
            f"ERC={w_erc[1]:.3f} vs EW={w_ew[1]:.3f}"
        )

    def test_erc_has_lower_max_rc_than_ew(self):
        """ERC should produce a more balanced risk budget than Equal Weight."""
        vols = [0.10, 0.50, 0.30, 0.20, 0.60]
        cov = cov_from_vols(vols, correlation=0.1)

        w_ew = equal_weight_portfolio(5)
        w_erc = equal_risk_contribution_portfolio(cov, max_weight=0.6)

        rc_ew = risk_contribution(w_ew, cov)
        rc_erc = risk_contribution(w_erc, cov)

        # Normalise to fractions for fair comparison
        rf_ew = rc_ew / rc_ew.sum()
        rf_erc = rc_erc / rc_erc.sum()

        max_rc_ew = rf_ew.max()
        max_rc_erc = rf_erc.max()

        assert max_rc_erc < max_rc_ew, (
            f"ERC max RC {max_rc_erc:.3f} should be < EW max RC {max_rc_ew:.3f}"
        )

    def test_erc_concentration_lower_than_ew_on_heterogeneous_universe(self):
        """In a heterogeneous-vol universe, ERC should have lower Herfindahl on RC."""
        vols = [0.10, 0.55, 0.10, 0.55, 0.10, 0.55]
        cov = cov_from_vols(vols, correlation=0.05)

        w_ew = equal_weight_portfolio(6)
        w_erc = equal_risk_contribution_portfolio(cov, max_weight=0.5)

        rc_ew = risk_contribution(w_ew, cov) / portfolio_volatility(w_ew, cov)
        rc_erc = risk_contribution(w_erc, cov) / portfolio_volatility(w_erc, cov)

        hhi_ew = (rc_ew ** 2).sum()
        hhi_erc = (rc_erc ** 2).sum()

        assert hhi_erc < hhi_ew, (
            f"ERC risk HHI {hhi_erc:.6f} should be < EW risk HHI {hhi_ew:.6f}"
        )

    def test_four_assets_extreme_vol_ratio(self):
        """4-asset case: vols differ by 5× — ERC weight ordering must invert vol ordering."""
        vols = [0.08, 0.40, 0.10, 0.35]
        cov = cov_from_vols(vols, correlation=0.0)
        w = equal_risk_contribution_portfolio(cov, max_weight=1.0)
        # Lowest-vol asset (index 0) should have highest weight
        assert w[0] == w.max() or abs(w[0] - w.max()) < 1e-4, (
            f"Lowest-vol asset should dominate ERC weights: {w}"
        )
        # Highest-vol asset (index 1) should have the lowest weight
        assert w[1] == w.min() or abs(w[1] - w.min()) < 1e-4, (
            f"Highest-vol asset should have lowest ERC weight: {w}"
        )


# ---------------------------------------------------------------------------
# 4 — ERC under weight constraints
# ---------------------------------------------------------------------------

class TestERCWithConstraints:

    def test_erc_with_tight_cap_still_balanced(self):
        """Even when max_weight is tight, risk contributions should be reasonably balanced."""
        # With 4 assets and max_weight=0.30, each asset is forced between 10%–30%
        vols = [0.10, 0.50, 0.25, 0.20]
        cov = cov_from_vols(vols, correlation=0.2)
        w = equal_risk_contribution_portfolio(cov, max_weight=0.30)

        # All weights in [0, 0.30 + eps]
        assert (w >= -1e-8).all()
        assert w.max() <= 0.30 + 1e-6
        assert abs(w.sum() - 1.0) < 1e-6

    def test_erc_with_20_pct_cap_universe(self):
        """Reproduce the main project constraint: 20 assets, 20% max weight."""
        vols = np.linspace(0.10, 0.65, 20).tolist()
        cov = cov_from_vols(vols, correlation=0.2)
        w = equal_risk_contribution_portfolio(cov, max_weight=0.20)

        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-8).all()
        assert w.max() <= 0.20 + 1e-6

        # High-vol assets (last ones) should have lower weights on average
        low_vol_mean = w[:5].mean()
        high_vol_mean = w[-5:].mean()
        assert low_vol_mean > high_vol_mean, (
            f"Low-vol mean weight {low_vol_mean:.4f} should exceed "
            f"high-vol mean weight {high_vol_mean:.4f}"
        )


# ---------------------------------------------------------------------------
# 5 — Walk-forward stability
# ---------------------------------------------------------------------------

class TestERCWalkForwardStability:

    def test_erc_walk_forward_no_fallback_on_clean_data(self):
        """On synthetic clean data, ERC walk-forward must never fall back to EW."""
        returns = synthetic_returns(n_assets=10, n_days=600, seed=7)
        _, _, meta = walk_forward_rebalance(
            returns=returns,
            optimizer_func=equal_risk_contribution_portfolio,
            optimizer_name="erc",
            lookback_window_days=120,
            rebalance_frequency="ME",
            transaction_cost_bps=15,
            max_weight=0.20,
        )
        assert not meta.empty
        assert meta["used_equal_weight_fallback"].sum() == 0, (
            "ERC should not fall back to EW on clean data"
        )

    def test_erc_weights_sum_to_one_at_each_rebalance(self):
        """Weights must sum to 1.0 at every rebalance window."""
        returns = synthetic_returns(n_assets=8, n_days=600, seed=11)
        _, wh, _ = walk_forward_rebalance(
            returns=returns,
            optimizer_func=equal_risk_contribution_portfolio,
            optimizer_name="erc",
            lookback_window_days=120,
            rebalance_frequency="ME",
            transaction_cost_bps=15,
            max_weight=0.20,
        )
        assert not wh.empty
        row_sums = wh.sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-6, (
            f"ERC weights do not sum to 1.0 at all rebalances; "
            f"max deviation = {(row_sums - 1.0).abs().max():.8f}"
        )

    def test_erc_weights_respect_cap_in_walk_forward(self):
        """The 20% cap must hold at every rebalance."""
        returns = synthetic_returns(
            n_assets=10, n_days=600,
            vols_ann=[0.10, 0.60, 0.10, 0.60, 0.10, 0.60, 0.10, 0.60, 0.10, 0.60],
            seed=3,
        )
        _, wh, _ = walk_forward_rebalance(
            returns=returns,
            optimizer_func=equal_risk_contribution_portfolio,
            optimizer_name="erc",
            lookback_window_days=120,
            rebalance_frequency="ME",
            transaction_cost_bps=15,
            max_weight=0.20,
        )
        assert not wh.empty
        max_w = wh.max().max()
        assert max_w <= 0.20 + 1e-5, (
            f"ERC weight cap violated in walk-forward: max weight = {max_w:.6f}"
        )

    def test_erc_walk_forward_heterogeneous_vols_overweights_low_vol(self):
        """Over the full walk-forward history, low-vol assets should have higher average ERC weight."""
        # 4 low-vol (10%) + 4 high-vol (55%) assets
        vols = [0.10, 0.10, 0.10, 0.10, 0.55, 0.55, 0.55, 0.55]
        returns = synthetic_returns(n_assets=8, n_days=600, vols_ann=vols, seed=5)
        _, wh, _ = walk_forward_rebalance(
            returns=returns,
            optimizer_func=equal_risk_contribution_portfolio,
            optimizer_name="erc",
            lookback_window_days=120,
            rebalance_frequency="ME",
            transaction_cost_bps=15,
            max_weight=0.30,
        )
        assert not wh.empty
        avg_weights = wh.mean()
        low_vol_avg = avg_weights.iloc[:4].mean()   # T0–T3 are low-vol
        high_vol_avg = avg_weights.iloc[4:].mean()  # T4–T7 are high-vol
        assert low_vol_avg > high_vol_avg + 0.03, (
            f"Low-vol avg weight {low_vol_avg:.4f} should exceed "
            f"high-vol avg weight {high_vol_avg:.4f} by a material margin"
        )
