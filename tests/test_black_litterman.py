"""
Tests for src/black_litterman.py — Black-Litterman portfolio optimization.

All tests use synthetic data only. No LSEG API calls required.
"""

import numpy as np
import pandas as pd
import pytest

from src.black_litterman import (
    BLView,
    BLResult,
    black_litterman_portfolio,
    black_litterman_posterior,
    bl_result_to_series,
    build_views,
    compare_bl_to_strategies,
    compute_implied_returns,
    compute_market_weights,
    get_ai_tech_views,
    run_black_litterman,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_universe(n: int = 5, seed: int = 42):
    """Small synthetic universe for fast tests."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n)]
    vols = rng.uniform(0.15, 0.45, n)
    corr = np.full((n, n), 0.4)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    mean_returns = rng.uniform(0.05, 0.35, n)
    return tickers, cov, mean_returns


def make_ai_tickers():
    """Real-ticker subset for views that reference actual tickers."""
    tickers = [
        "NVDA.O", "MSFT.O", "GOOGL.O", "AMD.O", "INTC.O",
        "AMZN.O", "META.O", "IBM.N", "TSM.N", "ASML.O",
    ]
    n = len(tickers)
    rng = np.random.default_rng(99)
    vols = rng.uniform(0.20, 0.50, n)
    corr = np.full((n, n), 0.5)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    return tickers, cov


# ---------------------------------------------------------------------------
# 1. compute_market_weights
# ---------------------------------------------------------------------------

class TestComputeMarketWeights:

    def test_equal_weight_when_no_caps(self):
        tickers = ["A", "B", "C", "D"]
        w = compute_market_weights(tickers)
        assert np.allclose(w, 0.25)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_cap_weighted_correct(self):
        tickers = ["A", "B"]
        caps = {"A": 3.0, "B": 1.0}
        w = compute_market_weights(tickers, caps)
        assert abs(w[0] - 0.75) < 1e-10
        assert abs(w[1] - 0.25) < 1e-10

    def test_missing_ticker_in_caps_gets_zero(self):
        tickers = ["A", "B", "C"]
        caps = {"A": 1.0, "B": 1.0}  # C missing
        w = compute_market_weights(tickers, caps)
        assert w[2] == 0.0
        assert abs(w.sum() - 1.0) < 1e-10

    def test_sum_to_one(self):
        tickers = ["A", "B", "C", "D", "E"]
        caps = {t: float(i + 1) for i, t in enumerate(tickers)}
        w = compute_market_weights(tickers, caps)
        assert abs(w.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# 2. compute_implied_returns
# ---------------------------------------------------------------------------

class TestComputeImpliedReturns:

    def test_output_shape(self):
        tickers, cov, _ = make_universe(5)
        w_mkt = np.ones(5) / 5
        Pi = compute_implied_returns(cov, w_mkt, risk_aversion=2.5)
        assert Pi.shape == (5,)

    def test_all_positive_with_positive_cov(self):
        """With positive covariance and positive weights, Pi should be positive."""
        tickers, cov, _ = make_universe(5)
        w_mkt = np.ones(5) / 5
        Pi = compute_implied_returns(cov, w_mkt, risk_aversion=2.5)
        assert (Pi > 0).all()

    def test_scales_with_risk_aversion(self):
        """Higher risk aversion → proportionally higher implied returns."""
        tickers, cov, _ = make_universe(5)
        w_mkt = np.ones(5) / 5
        Pi1 = compute_implied_returns(cov, w_mkt, risk_aversion=2.0)
        Pi2 = compute_implied_returns(cov, w_mkt, risk_aversion=4.0)
        assert np.allclose(Pi2, 2.0 * Pi1)

    def test_equal_weight_gives_symmetric_prior(self):
        """With EW market weights and symmetric cov, all assets get similar prior."""
        n = 4
        vols = np.full(n, 0.25)
        corr = np.full((n, n), 0.5)
        np.fill_diagonal(corr, 1.0)
        cov = np.outer(vols, vols) * corr
        w_mkt = np.ones(n) / n
        Pi = compute_implied_returns(cov, w_mkt, 2.5)
        # All Pi values should be equal (symmetric problem)
        assert np.allclose(Pi, Pi[0])


# ---------------------------------------------------------------------------
# 3. build_views
# ---------------------------------------------------------------------------

class TestBuildViews:

    def test_absolute_view_p_matrix(self):
        """Absolute view on asset 0: P[0] should be [1, 0, 0, ...]."""
        tickers = ["A", "B", "C"]
        _, cov, _ = make_universe(3)
        views = [BLView("A hits 20%", {"A": 1.0}, 0.20)]
        P, Q, Omega = build_views(views, tickers, cov, tau=0.05)
        assert P.shape == (1, 3)
        assert P[0, 0] == 1.0
        assert P[0, 1] == 0.0
        assert P[0, 2] == 0.0

    def test_relative_view_p_matrix(self):
        """Relative view A vs B: P[0] should be [1, -1, 0]."""
        tickers = ["A", "B", "C"]
        _, cov, _ = make_universe(3)
        views = [BLView("A beats B", {"A": 1.0, "B": -1.0}, 0.10)]
        P, Q, Omega = build_views(views, tickers, cov, tau=0.05)
        assert P[0, 0] == 1.0
        assert P[0, 1] == -1.0
        assert P[0, 2] == 0.0

    def test_q_vector_matches_expected_returns(self):
        tickers = ["A", "B", "C"]
        _, cov, _ = make_universe(3)
        views = [
            BLView("v1", {"A": 1.0}, 0.15),
            BLView("v2", {"B": 1.0, "C": -1.0}, 0.08),
        ]
        P, Q, Omega = build_views(views, tickers, cov, tau=0.05)
        assert Q[0] == 0.15
        assert Q[1] == 0.08

    def test_omega_is_diagonal(self):
        tickers, cov, _ = make_universe(4)
        views = [BLView("v1", {"T0": 1.0}, 0.20), BLView("v2", {"T1": 1.0, "T2": -1.0}, 0.10)]
        P, Q, Omega = build_views(views, tickers, cov, tau=0.05)
        off_diag = Omega - np.diag(np.diag(Omega))
        assert np.allclose(off_diag, 0.0)

    def test_omega_positive_diagonal(self):
        tickers, cov, _ = make_universe(4)
        views = [BLView("v1", {"T0": 1.0}, 0.20)]
        P, Q, Omega = build_views(views, tickers, cov, tau=0.05)
        assert (np.diag(Omega) > 0).all()

    def test_custom_confidence_overrides_default(self):
        tickers = ["A", "B"]
        _, cov, _ = make_universe(2)
        custom_conf = 0.001
        views = [BLView("v1", {"A": 1.0}, 0.20, confidence=custom_conf)]
        P, Q, Omega = build_views(views, tickers, cov, tau=0.05)
        assert abs(Omega[0, 0] - custom_conf) < 1e-12

    def test_unknown_ticker_in_view_is_silently_ignored(self):
        tickers = ["A", "B", "C"]
        _, cov, _ = make_universe(3)
        views = [BLView("v1", {"UNKNOWN": 1.0, "A": -1.0}, 0.10)]
        P, Q, Omega = build_views(views, tickers, cov, tau=0.05)
        # UNKNOWN ignored, A gets -1.0
        assert P[0, 0] == -1.0
        assert P[0, 1] == 0.0


# ---------------------------------------------------------------------------
# 4. black_litterman_posterior
# ---------------------------------------------------------------------------

class TestBlackLittermanPosterior:

    def _setup(self):
        tickers, cov, _ = make_universe(5)
        w_mkt = np.ones(5) / 5
        Pi = compute_implied_returns(cov, w_mkt, 2.5)
        views = [BLView("T0 at 30%", {"T0": 1.0}, 0.30)]
        P, Q, Omega = build_views(views, tickers, cov, tau=0.05)
        return cov, Pi, P, Q, Omega

    def test_output_shapes(self):
        cov, Pi, P, Q, Omega = self._setup()
        mu_BL, Sigma_BL = black_litterman_posterior(cov, Pi, P, Q, Omega, tau=0.05)
        n = len(Pi)
        assert mu_BL.shape == (n,)
        assert Sigma_BL.shape == (n, n)

    def test_posterior_cov_symmetric(self):
        cov, Pi, P, Q, Omega = self._setup()
        mu_BL, Sigma_BL = black_litterman_posterior(cov, Pi, P, Q, Omega, tau=0.05)
        assert np.allclose(Sigma_BL, Sigma_BL.T, atol=1e-10)

    def test_view_pulls_posterior_toward_view(self):
        """A bullish view on T0 should increase T0's posterior return vs prior."""
        tickers, cov, _ = make_universe(5)
        w_mkt = np.ones(5) / 5
        Pi = compute_implied_returns(cov, w_mkt, 2.5)
        views = [BLView("T0 bullish", {"T0": 1.0}, Pi[0] + 0.10)]
        P, Q, Omega = build_views(views, tickers, cov, tau=0.05)
        mu_BL, _ = black_litterman_posterior(cov, Pi, P, Q, Omega, tau=0.05)
        assert mu_BL[0] > Pi[0], "Bullish view should increase T0 posterior above prior"

    def test_no_views_posterior_equals_prior(self):
        """If views exactly match the prior, posterior should equal the prior."""
        tickers, cov, _ = make_universe(5)
        w_mkt = np.ones(5) / 5
        Pi = compute_implied_returns(cov, w_mkt, 2.5)
        # View exactly at the prior return for T0
        views = [BLView("view at prior", {"T0": 1.0}, Pi[0])]
        P, Q, Omega = build_views(views, tickers, cov, tau=0.05)
        mu_BL, _ = black_litterman_posterior(cov, Pi, P, Q, Omega, tau=0.05)
        # Posterior for T0 should stay very close to prior when view == prior
        assert abs(mu_BL[0] - Pi[0]) < 0.05


# ---------------------------------------------------------------------------
# 5. black_litterman_portfolio
# ---------------------------------------------------------------------------

class TestBlackLittermanPortfolio:

    def test_weights_sum_to_one(self):
        tickers, cov, mean_returns = make_universe(5)
        w = black_litterman_portfolio(mean_returns, cov, 0.04, 0.3)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self):
        tickers, cov, mean_returns = make_universe(5)
        w = black_litterman_portfolio(mean_returns, cov, 0.04, 0.3)
        assert (w >= -1e-8).all()

    def test_max_weight_constraint(self):
        for max_w in [0.20, 0.25, 0.40, 1.0]:
            tickers, cov, mean_returns = make_universe(5)
            w = black_litterman_portfolio(mean_returns, cov, 0.04, max_w)
            assert w.max() <= max_w + 1e-6, f"Max weight violated for cap={max_w}"

    def test_high_return_asset_gets_high_weight(self):
        """Asset with highest expected return should get a significant allocation."""
        n = 5
        vols = np.full(n, 0.25)
        cov = np.diag(vols ** 2)  # zero correlation for clean test
        mean_returns = np.array([0.05, 0.05, 0.35, 0.05, 0.05])  # T2 dominates
        w = black_litterman_portfolio(mean_returns, cov, 0.04, 0.5)
        assert w[2] == max(w), "Highest-return asset should get highest weight"


# ---------------------------------------------------------------------------
# 6. run_black_litterman (full pipeline)
# ---------------------------------------------------------------------------

class TestRunBlackLitterman:

    def test_returns_bl_result(self):
        tickers, cov = make_ai_tickers()
        views = [BLView("NVDA at 40%", {"NVDA.O": 1.0}, 0.40)]
        result = run_black_litterman(tickers, cov, views)
        assert isinstance(result, BLResult)

    def test_weights_sum_to_one(self):
        tickers, cov = make_ai_tickers()
        views = get_ai_tech_views("medium")
        result = run_black_litterman(tickers, cov, views)
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self):
        tickers, cov = make_ai_tickers()
        views = get_ai_tech_views("medium")
        result = run_black_litterman(tickers, cov, views)
        assert (result.weights >= -1e-8).all()

    def test_tickers_preserved(self):
        tickers, cov = make_ai_tickers()
        views = [BLView("NVDA at 40%", {"NVDA.O": 1.0}, 0.40)]
        result = run_black_litterman(tickers, cov, views)
        assert result.tickers == tickers

    def test_bullish_view_increases_posterior(self):
        """Bullish absolute view on NVDA should pull its posterior above its prior."""
        tickers, cov = make_ai_tickers()
        nvda_idx = tickers.index("NVDA.O")
        views = [BLView("NVDA bullish", {"NVDA.O": 1.0}, 0.50)]
        result = run_black_litterman(tickers, cov, views, risk_aversion=2.5)
        assert result.posterior_returns[nvda_idx] > result.equilibrium_returns[nvda_idx]

    def test_all_three_views(self):
        """Full pipeline with all 3 AI/Tech views should run without error."""
        tickers, cov = make_ai_tickers()
        views = get_ai_tech_views("medium")
        result = run_black_litterman(tickers, cov, views, max_weight=0.20)
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert result.weights.max() <= 0.20 + 1e-6

    def test_higher_confidence_moves_posterior_more(self):
        """High confidence views should pull the posterior further from the prior."""
        tickers, cov = make_ai_tickers()
        nvda_idx = tickers.index("NVDA.O")
        views_low = get_ai_tech_views("low")
        views_high = get_ai_tech_views("high")
        result_low = run_black_litterman(tickers, cov, views_low)
        result_high = run_black_litterman(tickers, cov, views_high)
        delta_low = abs(result_low.posterior_returns[nvda_idx] - result_low.equilibrium_returns[nvda_idx])
        delta_high = abs(result_high.posterior_returns[nvda_idx] - result_high.equilibrium_returns[nvda_idx])
        assert delta_high > delta_low, "Higher confidence should pull posterior further from prior"


# ---------------------------------------------------------------------------
# 7. Utility functions
# ---------------------------------------------------------------------------

class TestBLUtilities:

    def test_bl_result_to_series(self):
        tickers, cov = make_ai_tickers()
        views = [BLView("test", {"NVDA.O": 1.0}, 0.30)]
        result = run_black_litterman(tickers, cov, views)
        s = bl_result_to_series(result)
        assert isinstance(s, pd.Series)
        assert list(s.index) == tickers
        assert abs(s.sum() - 1.0) < 1e-6

    def test_compare_bl_to_strategies(self):
        tickers, cov = make_ai_tickers()
        views = get_ai_tech_views("medium")
        result = run_black_litterman(tickers, cov, views)
        n = len(tickers)
        other = {"EW": pd.Series(np.ones(n)/n, index=tickers)}
        mean_ret = np.full(n, 0.15)
        df = compare_bl_to_strategies(result, other, cov, mean_ret, 0.04)
        assert "Black-Litterman" in df.index
        assert "EW" in df.index
        assert "Sharpe Ratio" in df.columns
        assert "Effective N" in df.columns

    def test_get_ai_tech_views_all_confidence_levels(self):
        for level in ["low", "medium", "high"]:
            views = get_ai_tech_views(level)
            assert len(views) == 3
            for v in views:
                assert isinstance(v, BLView)
                assert v.expected_return > 0

    def test_export_bl_results(self, tmp_path):
        tickers, cov = make_ai_tickers()
        views = get_ai_tech_views("medium")
        result = run_black_litterman(tickers, cov, views)
        export_path = str(tmp_path)

        from src.black_litterman import export_bl_results
        export_bl_results(result, output_dir=export_path)

        import os
        assert os.path.exists(f"{export_path}/bl_weights_v1.csv")
        assert os.path.exists(f"{export_path}/bl_returns_v1.csv")
        assert os.path.exists(f"{export_path}/bl_views_v1.csv")
