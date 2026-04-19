"""Black-Litterman portfolio optimization module.

Implements the Black-Litterman (1990) model for blending market equilibrium
returns with analyst views into posterior expected returns, which are then
used to construct an optimal portfolio.

The model solves:

    mu_BL = [(tau*Sigma)^{-1} + P' Omega^{-1} P]^{-1}
            * [(tau*Sigma)^{-1} * Pi + P' Omega^{-1} * Q]

where:
    Pi    : implied equilibrium returns = lambda * Sigma @ w_mkt
    P     : (K x n) view matrix — rows define views on assets
    Q     : (K,) vector of view expected returns
    Omega : (K x K) diagonal view uncertainty matrix
    tau   : scalar confidence in the prior (typically 0.01–0.05)
    lambda: market risk aversion coefficient

References:
    Black, F. & Litterman, R. (1990). Asset Allocation: Combining Investor
    Views with Market Equilibrium. Goldman Sachs Fixed Income Research.

    He, G. & Litterman, R. (1999). The Intuition Behind Black-Litterman
    Model Portfolios. Goldman Sachs Investment Management Research.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BLView:
    """A single analyst view in Black-Litterman format.

    Attributes
    ----------
    name:
        Human-readable description of the view (e.g. "NVDA outperforms INTC").
    assets:
        Dict mapping ticker -> weight in the view portfolio.
        Absolute view: {ticker: 1.0}
        Relative view: {winner: 1.0, loser: -1.0}
        Sector view: {ticker_A: 0.5, ticker_B: 0.5, ticker_C: -0.5, ...}
    expected_return:
        Annualized return expected by the analyst for this view (e.g. 0.15 for 15%).
    confidence:
        Optional override for view uncertainty. If None, uses the standard
        Omega = tau * P @ Sigma @ P.T (proportional to prior uncertainty).
        If provided, directly sets the diagonal element of Omega for this view.
    """
    name: str
    assets: Dict[str, float]
    expected_return: float
    confidence: Optional[float] = None


@dataclass
class BLResult:
    """Output of the Black-Litterman computation.

    Attributes
    ----------
    tickers:
        Asset names, aligned with all vector outputs.
    equilibrium_returns:
        Pi — implied market equilibrium expected returns (annualized).
    posterior_returns:
        mu_BL — posterior expected returns after incorporating views (annualized).
    posterior_cov:
        Sigma_BL — posterior covariance matrix (annualized).
    weights:
        Optimal portfolio weights from Max Sharpe on posterior returns.
    views_used:
        List of BLView objects that were used.
    lambda_:
        Risk aversion coefficient used.
    tau:
        Confidence in prior used.
    """
    tickers: List[str]
    equilibrium_returns: np.ndarray
    posterior_returns: np.ndarray
    posterior_cov: np.ndarray
    weights: np.ndarray
    views_used: List[BLView]
    lambda_: float
    tau: float


# ---------------------------------------------------------------------------
# Core Black-Litterman functions
# ---------------------------------------------------------------------------

def compute_market_weights(
    tickers: List[str],
    market_caps: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Compute market portfolio weights for the universe.

    If market_caps is provided, returns cap-weighted portfolio.
    Otherwise falls back to equal weights (documented assumption).

    Parameters
    ----------
    tickers:
        List of asset tickers.
    market_caps:
        Optional dict mapping ticker -> market cap (any unit).

    Returns
    -------
    np.ndarray of weights summing to 1.0.
    """
    n = len(tickers)

    if market_caps is None:
        # Documented assumption: equal-weight proxy for market portfolio.
        # This is a conservative, defensible choice when cap data is unavailable.
        # It means Pi will be symmetric — all assets start at the same prior.
        return np.ones(n) / n

    caps = np.array([market_caps.get(t, 0.0) for t in tickers], dtype=float)
    total = caps.sum()
    if total <= 0:
        return np.ones(n) / n
    return caps / total


def compute_implied_returns(
    cov_matrix: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """Compute implied equilibrium expected returns (Pi).

    Pi = lambda * Sigma * w_mkt

    This is the return vector that makes the market portfolio mean-variance
    optimal. It represents the "prior" — what the market as a whole expects.

    Parameters
    ----------
    cov_matrix:
        (n x n) annualized covariance matrix.
    market_weights:
        (n,) market portfolio weights.
    risk_aversion:
        Lambda — market risk aversion coefficient.
        Typical range: 2.0 to 4.0.
        Can be back-calculated as: lambda = (E[r_mkt] - rf) / Var[r_mkt]
        With Sharpe ~0.5, market vol ~20%: lambda = 0.5/0.20 = 2.5

    Returns
    -------
    (n,) annualized implied equilibrium returns.
    """
    return risk_aversion * cov_matrix @ market_weights


def build_views(
    views: List[BLView],
    tickers: List[str],
    cov_matrix: np.ndarray,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the P, Q, Omega matrices from a list of BLView objects.

    Parameters
    ----------
    views:
        List of analyst views.
    tickers:
        Full list of asset tickers (defines column order of P).
    cov_matrix:
        (n x n) annualized covariance matrix (used for default Omega).
    tau:
        Prior confidence scalar.

    Returns
    -------
    P : ndarray (K x n)
        View matrix. Each row defines one view as a portfolio of assets.
    Q : ndarray (K,)
        Vector of expected returns for each view (annualized).
    Omega : ndarray (K x K)
        Diagonal view uncertainty matrix.
        Default: Omega_kk = tau * p_k' * Sigma * p_k
        (view is as uncertain as the prior uncertainty in that direction)
    """
    n = len(tickers)
    ticker_index = {t: i for i, t in enumerate(tickers)}
    K = len(views)

    P = np.zeros((K, n))
    Q = np.zeros(K)

    for k, view in enumerate(views):
        for ticker, weight in view.assets.items():
            if ticker in ticker_index:
                P[k, ticker_index[ticker]] = weight
            # Silently skip tickers not in the universe
        Q[k] = view.expected_return

    # Default Omega: proportional to prior uncertainty
    # This is the "neutral" assumption — views carry same uncertainty as the prior
    raw_omega = tau * P @ cov_matrix @ P.T
    omega_diag = np.diag(raw_omega).copy()  # copy to make writable

    # Override individual view uncertainties if confidence is specified
    for k, view in enumerate(views):
        if view.confidence is not None:
            omega_diag[k] = view.confidence

    Omega = np.diag(omega_diag)
    return P, Q, Omega


def black_litterman_posterior(
    cov_matrix: np.ndarray,
    equilibrium_returns: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Black-Litterman posterior returns and covariance.

    Formula (He & Litterman, 1999):

        M = (tau*Sigma)^{-1} + P' Omega^{-1} P
        mu_BL = M^{-1} * [(tau*Sigma)^{-1} Pi + P' Omega^{-1} Q]
        Sigma_BL = Sigma + M^{-1}

    The posterior mean blends the equilibrium prior (Pi) with analyst views (Q),
    weighted by their respective uncertainties.

    Parameters
    ----------
    cov_matrix:
        (n x n) annualized covariance matrix Sigma.
    equilibrium_returns:
        (n,) implied equilibrium returns Pi.
    P:
        (K x n) view matrix.
    Q:
        (K,) view expected returns.
    Omega:
        (K x K) diagonal view uncertainty matrix.
    tau:
        Prior confidence scalar.

    Returns
    -------
    mu_BL : ndarray (n,)
        Posterior expected returns.
    Sigma_BL : ndarray (n x n)
        Posterior covariance matrix.
    """
    tau_Sigma = tau * cov_matrix
    tau_Sigma_inv = np.linalg.inv(tau_Sigma + np.eye(len(cov_matrix)) * 1e-10)
    Omega_inv = np.linalg.inv(Omega + np.eye(len(Omega)) * 1e-12)

    # Precision matrix
    M = tau_Sigma_inv + P.T @ Omega_inv @ P

    # Posterior mean
    b = tau_Sigma_inv @ equilibrium_returns + P.T @ Omega_inv @ Q
    mu_BL = np.linalg.solve(M, b)

    # Posterior covariance
    Sigma_BL = cov_matrix + np.linalg.inv(M)
    Sigma_BL = (Sigma_BL + Sigma_BL.T) / 2  # enforce symmetry

    return mu_BL, Sigma_BL


def black_litterman_portfolio(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    max_weight: float,
) -> np.ndarray:
    """Compute optimal (Max Sharpe) portfolio on BL posterior returns.

    Parameters
    ----------
    mean_returns:
        (n,) posterior BL expected returns (mu_BL).
    cov_matrix:
        (n x n) covariance matrix for optimization.
        Typically the original Sigma (not Sigma_BL), since BL primarily
        updates the mean, not the covariance structure.
    risk_free_rate:
        Annualized risk-free rate.
    max_weight:
        Maximum weight per asset (long-only cap).

    Returns
    -------
    (n,) optimal portfolio weights summing to 1.
    """
    n = len(mean_returns)
    w0 = np.ones(n) / n

    def neg_sharpe(w: np.ndarray) -> float:
        ret = float(w @ mean_returns)
        vol = float(np.sqrt(np.clip(w @ cov_matrix @ w, 0, None)))
        if vol < 1e-10:
            return 1e6
        return -(ret - risk_free_rate) / vol

    result = minimize(
        fun=neg_sharpe,
        x0=w0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not result.success:
        return w0

    weights = result.x
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    return weights / total if total > 0 else w0


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def run_black_litterman(
    tickers: List[str],
    cov_matrix: np.ndarray,
    views: List[BLView],
    risk_free_rate: float = 0.04,
    max_weight: float = 0.20,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    market_caps: Optional[Dict[str, float]] = None,
    returns_for_prior: Optional["pd.DataFrame"] = None,
    prior_cov_method: str = "as_given",
) -> BLResult:
    """Full Black-Litterman pipeline: equilibrium → views → posterior → weights.

    Parameters
    ----------
    tickers:
        List of asset tickers (must match cov_matrix column/row order).
    cov_matrix:
        (n x n) annualized covariance matrix. Used for optimization (Step 4).
    views:
        List of BLView objects defining analyst views.
    risk_free_rate:
        Annualized risk-free rate.
    max_weight:
        Maximum weight per asset (long-only cap).
    risk_aversion:
        Lambda — market risk aversion (default 2.5).
    tau:
        Confidence in the equilibrium prior (default 0.05).
    market_caps:
        Optional cap-weights for the market portfolio.
        If None, equal-weight proxy is used (documented assumption).
    returns_for_prior:
        Optional DataFrame of daily returns. Required when prior_cov_method != "as_given".
        Used to re-estimate the prior covariance (for Pi and Omega) using a different
        estimator than the one used for portfolio optimization.
    prior_cov_method:
        "as_given"    → use cov_matrix as both prior and optimization covariance (default)
        "factor"      → re-estimate prior covariance using factor model (Σ = BFBᵀ + D)
        "ledoit_wolf" → re-estimate prior covariance using Ledoit-Wolf shrinkage
        The prior covariance affects Pi (implied returns) and Omega (view uncertainty).
        The portfolio optimization always uses the original cov_matrix.

    Returns
    -------
    BLResult object with all intermediate and final outputs.
    """
    n = len(tickers)
    assert cov_matrix.shape == (n, n), f"cov_matrix shape {cov_matrix.shape} != ({n},{n})"
    assert len(views) > 0, "At least one view is required"

    # Determine which covariance to use for the prior (Pi and Omega)
    if prior_cov_method == "as_given" or returns_for_prior is None:
        prior_cov = cov_matrix
    else:
        from src.covariance import build_covariance_matrix
        # Subset returns to valid tickers in case of partial overlap
        valid_cols = [t for t in tickers if t in returns_for_prior.columns]
        if len(valid_cols) == n:
            prior_cov = build_covariance_matrix(
                returns_for_prior[tickers], method=prior_cov_method
            )
        else:
            # Fallback: use as-given if column mismatch
            prior_cov = cov_matrix

    # Step 1: Market weights and implied equilibrium returns (using prior_cov)
    w_mkt = compute_market_weights(tickers, market_caps)
    Pi = compute_implied_returns(prior_cov, w_mkt, risk_aversion)

    # Step 2: Build view matrices (Omega uses prior_cov for proportionality)
    P, Q, Omega = build_views(views, tickers, prior_cov, tau)

    # Step 3: Posterior returns and covariance (using prior_cov for precision matrix)
    mu_BL, Sigma_BL = black_litterman_posterior(
        prior_cov, Pi, P, Q, Omega, tau
    )

    # Step 4: Optimal portfolio on posterior returns
    # Always use the original cov_matrix for optimization — BL updates the mean
    weights = black_litterman_portfolio(mu_BL, cov_matrix, risk_free_rate, max_weight)

    return BLResult(
        tickers=tickers,
        equilibrium_returns=Pi,
        posterior_returns=mu_BL,
        posterior_cov=Sigma_BL,
        weights=weights,
        views_used=views,
        lambda_=risk_aversion,
        tau=tau,
    )


def generate_momentum_views(
    returns: pd.DataFrame,
    tickers: List[str],
    n_longs: int = 3,
    n_shorts: int = 3,
    spread: float = 0.10,
    lookback_days: int = 126,
) -> List[BLView]:
    """Generate momentum-based relative views for walk-forward BL.

    At each rebalance window, ranks all valid assets by their recent momentum
    (past `lookback_days` cumulative return) and creates one relative view:
    the top-N momentum assets are expected to outperform the bottom-N by
    `spread` (annualised), with equal weights on each side.

    This is a simple, rule-based, non-discretionary view generation approach.
    It is look-ahead free: momentum is computed only from the training window.

    Design choices:
    - Uses 126-day (6-month) lookback — standard cross-sectional momentum
    - Single relative view keeps Omega 1×1 — minimal risk of ill-conditioning
    - Spread fixed at 10% — calibrated to cross-sectional momentum premium
      documented in Jegadeesh & Titman (1993) and subsequent literature

    Parameters
    ----------
    returns:
        DataFrame of valid asset daily returns in the training window.
        Must contain at least (n_longs + n_shorts) columns.
    tickers:
        Ordered list of valid asset tickers (= returns.columns).
    n_longs:
        Number of high-momentum assets on the long side of the view.
    n_shorts:
        Number of low-momentum assets on the short side.
    spread:
        Annualised expected outperformance of longs vs shorts (default 0.10 = 10%).
    lookback_days:
        Days to compute momentum. If training window is shorter, uses full window.

    Returns
    -------
    List with exactly one BLView (relative momentum view).
    Returns empty list if fewer than (n_longs + n_shorts) valid assets.
    """
    n = len(tickers)
    if n < n_longs + n_shorts:
        return []

    # Momentum = cumulative return over available lookback
    days = min(lookback_days, len(returns) - 1)
    if days <= 0:
        return []

    window = returns.iloc[-days:]
    cumulative = (1 + window).prod() - 1  # Series indexed by ticker

    ranked = cumulative.sort_values(ascending=False)
    longs = ranked.index[:n_longs].tolist()
    shorts = ranked.index[-n_shorts:].tolist()

    long_weight = 1.0 / n_longs
    short_weight = -1.0 / n_shorts

    assets = {t: long_weight for t in longs}
    assets.update({t: short_weight for t in shorts})

    long_names = ", ".join(longs[:2]) + ("..." if n_longs > 2 else "")
    short_names = ", ".join(shorts[:2]) + ("..." if n_shorts > 2 else "")
    view_name = f"Momentum: [{long_names}] +{spread:.0%} vs [{short_names}]"

    return [BLView(
        name=view_name,
        assets=assets,
        expected_return=spread,
        confidence=None,  # use default Omega
    )]


def black_litterman_optimizer(
    valid_train_slice: "pd.DataFrame",
    cov_matrix: np.ndarray,
    mean_returns: np.ndarray,
    valid_assets: List[str],
    risk_free_rate: float,
    max_weight: float,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> np.ndarray:
    """Walk-forward BL optimizer callable for the rebalancer engine.

    Generates momentum views from the training window, runs Black-Litterman,
    and returns the optimal weight array. Designed to be called inside
    walk_forward_rebalance via optimizer_name='black_litterman'.

    Falls back to equal weight if view generation fails or BL is ill-posed.

    Parameters
    ----------
    valid_train_slice:
        Returns DataFrame for valid assets in the lookback window.
    cov_matrix:
        Pre-computed annualized covariance matrix (n x n).
    mean_returns:
        Pre-computed annualized mean returns (n,).
    valid_assets:
        Ordered list of valid asset tickers.
    risk_free_rate:
        Annualized risk-free rate.
    max_weight:
        Maximum weight per asset.
    risk_aversion:
        BL lambda parameter.
    tau:
        BL tau parameter.

    Returns
    -------
    np.ndarray of weights summing to 1.
    """
    n = len(valid_assets)
    fallback = np.ones(n) / n

    # Generate momentum views from training data
    views = generate_momentum_views(
        returns=valid_train_slice,
        tickers=valid_assets,
    )
    if not views:
        return fallback

    try:
        result = run_black_litterman(
            tickers=valid_assets,
            cov_matrix=cov_matrix,
            views=views,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            risk_aversion=risk_aversion,
            tau=tau,
        )
        return result.weights
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Predefined views for the AI/Tech universe (static BL only)
# ---------------------------------------------------------------------------

def get_ai_tech_views(confidence_level: str = "medium") -> List[BLView]:
    """Return a set of analyst views for the AI/Tech universe.

    Three views are defined, ranging from individual stock to sector level.
    They are coherent with the AI/semiconductor supercycle thesis (2024-2026).

    Parameters
    ----------
    confidence_level:
        "low"    → high Omega uncertainty (views barely move the prior)
        "medium" → standard Omega (proportional to prior uncertainty)
        "high"   → low Omega uncertainty (views dominate the prior)
        Note: "medium" uses the default Omega = tau * P Sigma P'. The
        "low"/"high" overrides use direct confidence values on each view.

    Returns
    -------
    List of BLView objects.
    """
    # Confidence overrides (direct diagonal Omega values, annualized variance units)
    confidence_map = {
        "low":    {"absolute": 0.025, "relative": 0.020, "sector": 0.015},
        "medium": {"absolute": None,  "relative": None,  "sector": None},   # use default
        "high":   {"absolute": 0.002, "relative": 0.001, "sector": 0.001},
    }
    conf = confidence_map.get(confidence_level, confidence_map["medium"])

    views = [
        # --- View 1: Absolute — NVDA delivers 40% annualized ---
        # Rationale: NVIDIA dominates AI chip supply. Data center revenue grew
        # 400%+ in 2024. Pricing power sustained through 2026 with H100/H200 cycle.
        BLView(
            name="NVDA: AI chip supercycle — 40% annualized return",
            assets={"NVDA.O": 1.0},
            expected_return=0.40,
            confidence=conf["absolute"],
        ),

        # --- View 2: Relative — NVDA outperforms INTC by 20% ---
        # Rationale: Structural divergence. Intel losing data center share to AMD/NVDA,
        # struggling with foundry spin-off and product delays. NVDA gains.
        BLView(
            name="NVDA outperforms INTC by 20% (AI leader vs legacy challenger)",
            assets={"NVDA.O": 1.0, "INTC.O": -1.0},
            expected_return=0.20,
            confidence=conf["relative"],
        ),

        # --- View 3: Sector — Semiconductors beat Cloud/Enterprise by 10% ---
        # Rationale: Semiconductor capex cycle driven by AI demand.
        # Cloud names already re-rated on AI; semis still have upside from
        # supply constraints and new product cycles (H100→H200→Blackwell).
        BLView(
            name="Semis (NVDA+AMD+TSM+ASML) outperform Cloud (MSFT+GOOGL+AMZN+IBM) by 10%",
            assets={
                "NVDA.O": 0.25, "AMD.O": 0.25, "TSM.N": 0.25, "ASML.O": 0.25,
                "MSFT.O": -0.25, "GOOGL.O": -0.25, "AMZN.O": -0.25, "IBM.N": -0.25,
            },
            expected_return=0.10,
            confidence=conf["sector"],
        ),
    ]
    return views


# ---------------------------------------------------------------------------
# Output and comparison utilities
# ---------------------------------------------------------------------------

def bl_result_to_series(result: BLResult) -> pd.Series:
    """Convert BL weights to a labeled pd.Series."""
    return pd.Series(result.weights, index=result.tickers, name="Black-Litterman")


def print_bl_summary(result: BLResult) -> None:
    """Print a clean summary of BL inputs and outputs to the terminal."""
    n = len(result.tickers)
    width = 76

    print("\n" + "=" * width)
    print("BLACK-LITTERMAN PORTFOLIO SUMMARY")
    print("=" * width)
    print(f"  Lambda (risk aversion) : {result.lambda_:.2f}")
    print(f"  Tau (prior confidence) : {result.tau:.3f}")
    print(f"  Views used             : {len(result.views_used)}")
    print()

    print("  ANALYST VIEWS:")
    for i, view in enumerate(result.views_used, 1):
        print(f"    {i}. {view.name}")
        print(f"       Expected return: {view.expected_return:+.1%}")

    print()
    print(f"  {'Ticker':<10} {'Prior (Pi)':>12} {'Posterior':>12} {'Delta':>10} {'Weight':>10}")
    print(f"  {'-'*56}")
    for i, ticker in enumerate(result.tickers):
        pi = result.equilibrium_returns[i]
        mu = result.posterior_returns[i]
        w = result.weights[i]
        marker = " ◄" if w > 0.001 else ""
        print(
            f"  {ticker:<10} {pi:>+11.2%} {mu:>+11.2%} {mu-pi:>+9.2%} "
            f"{w:>9.2%}{marker}"
        )

    print()
    port_ret = result.weights @ result.posterior_returns
    port_vol = np.sqrt(result.weights @ result.posterior_cov[:len(result.tickers),
                                                              :len(result.tickers)] @ result.weights)
    print(f"  Portfolio expected return (BL) : {port_ret:+.2%}")
    print(f"  Portfolio expected vol  (BL)   : {port_vol:.2%}")
    print("=" * width + "\n")


def compare_bl_to_strategies(
    bl_result: BLResult,
    other_weights: Dict[str, pd.Series],
    cov_matrix: np.ndarray,
    mean_returns: np.ndarray,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Compare BL portfolio to other strategies on key metrics.

    Parameters
    ----------
    bl_result:
        Output of run_black_litterman.
    other_weights:
        Dict mapping strategy name to weight Series (indexed by ticker).
    cov_matrix:
        Covariance matrix for vol computation.
    mean_returns:
        Sample mean returns (for expected return computation).
    risk_free_rate:
        Annualized risk-free rate.

    Returns
    -------
    DataFrame with one row per strategy and columns:
    Expected Return, Volatility, Sharpe Ratio, Max Weight, Effective N.
    """
    rows = []
    tickers = bl_result.tickers

    # Include BL
    all_strategies = {"Black-Litterman": bl_result_to_series(bl_result)}
    all_strategies.update(other_weights)

    for name, weights_series in all_strategies.items():
        w = weights_series.reindex(tickers).fillna(0.0).values
        w = w / w.sum() if w.sum() > 0 else w

        exp_ret = float(w @ mean_returns)
        vol = float(np.sqrt(np.clip(w @ cov_matrix @ w, 0, None)))
        sharpe = (exp_ret - risk_free_rate) / vol if vol > 0 else np.nan
        max_w = float(w.max())
        eff_n = float(1.0 / (w ** 2).sum()) if (w > 0).any() else np.nan

        rows.append({
            "Strategy": name,
            "Exp. Return": exp_ret,
            "Volatility": vol,
            "Sharpe Ratio": sharpe,
            "Max Weight": max_w,
            "Effective N": eff_n,
        })

    df = pd.DataFrame(rows).set_index("Strategy")
    return df


def export_bl_results(
    result: BLResult,
    output_dir: str = "output/reports",
) -> None:
    """Export BL weights, posterior returns, and view summary to CSV.

    Parameters
    ----------
    result:
        Output of run_black_litterman.
    output_dir:
        Directory to write files into.
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Weights
    pd.Series(result.weights, index=result.tickers, name="weight").to_csv(
        output_dir / "bl_weights_v1.csv", header=True
    )

    # Prior vs posterior returns
    pd.DataFrame({
        "ticker": result.tickers,
        "equilibrium_return": result.equilibrium_returns,
        "posterior_return": result.posterior_returns,
        "delta": result.posterior_returns - result.equilibrium_returns,
    }).set_index("ticker").to_csv(output_dir / "bl_returns_v1.csv")

    # Views summary
    view_rows = []
    for v in result.views_used:
        view_rows.append({
            "view_name": v.name,
            "expected_return": v.expected_return,
            "assets": str(v.assets),
        })
    pd.DataFrame(view_rows).to_csv(output_dir / "bl_views_v1.csv", index=False)
