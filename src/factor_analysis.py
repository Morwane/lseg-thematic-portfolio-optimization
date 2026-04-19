"""Factor exposure analysis module for understanding portfolio positioning."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Sector mapping: each ticker to its primary sector
SECTOR_MAPPING = {
    "NVDA.O": "Semiconductors",
    "AMD.O": "Semiconductors",
    "INTC.O": "Semiconductors",
    "QCOM.O": "Semiconductors",
    "ARM.O": "Semiconductors",
    "TSM.N": "Semiconductors",
    "ASML.O": "Semiconductors",
    "MSFT.O": "Cloud/Enterprise",
    "GOOGL.O": "Cloud/Enterprise",
    "AMZN.O": "Cloud/Enterprise",
    "ORCL.N": "Cloud/Enterprise",
    "CRM.N": "Cloud/Enterprise",
    "IBM.N": "Cloud/Enterprise",
    "ADBE.O": "Cloud/Enterprise",
    "META.O": "Social/AI",
    "AAPL.O": "Consumer",
    "NOW.N": "Software",
    "SNOW.N": "Software",
    "PLTR.N": "Software",
    "MSTR.O": "Software",
}

TRADING_DAYS = 252


def compute_rolling_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling beta of stock relative to market.
    
    Args:
        stock_returns: Series of daily returns for one stock
        market_returns: Series of daily returns for market proxy
        window: Rolling window in days
        
    Returns:
        Series of rolling betas aligned with stock_returns index
    """
    # Align indices
    aligned = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns,
    }).dropna()
    
    betas = []
    for i in range(len(aligned)):
        if i < window:
            betas.append(np.nan)
        else:
            window_stock = aligned['stock'].iloc[i-window:i].values
            window_market = aligned['market'].iloc[i-window:i].values
            
            # Covariance / variance of market
            covariance = np.cov(window_stock, window_market)[0, 1]
            market_var = np.var(window_market)
            
            if market_var > 1e-8:
                beta = covariance / market_var
            else:
                beta = np.nan
            
            betas.append(beta)
    
    beta_series = pd.Series(betas, index=aligned.index)
    return beta_series


def compute_momentum_scores(
    prices: pd.Series,
    lookback_months: int = 12,
) -> pd.Series:
    """Compute 6-month momentum scores (12-month lookback minus 1-month).
    
    Args:
        prices: Series of daily prices for one stock
        lookback_months: Total lookback period (typically 12)
        
    Returns:
        Series of momentum scores (-1 to +1 range)
    """
    # Convert months to days approximately
    days_12m = lookback_months * 21
    days_1m = 21
    
    momentum_scores = []
    for i in range(len(prices)):
        if i < days_12m:
            momentum_scores.append(np.nan)
        else:
            price_12m_ago = prices.iloc[i - days_12m]
            price_1m_ago = prices.iloc[i - days_1m]
            price_now = prices.iloc[i]
            
            # 12-month return
            ret_12m = (price_1m_ago - price_12m_ago) / price_12m_ago
            # 1-month return
            ret_1m = (price_now - price_1m_ago) / price_1m_ago
            
            # Momentum = 12m return (skip last month to avoid look-ahead bias)
            momentum = ret_12m
            
            momentum_scores.append(momentum)
    
    momentum_series = pd.Series(momentum_scores, index=prices.index)
    return momentum_series


def compute_volatility_scores(
    returns: pd.Series,
    window: int = 126,  # ~6 months
) -> pd.Series:
    """Compute rolling realized volatility (annualized).
    
    Args:
        returns: Series of daily returns
        window: Rolling window in days
        
    Returns:
        Series of annualized volatility
    """
    rolling_std = returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS)
    return rolling_std


def compute_market_factor(
    all_returns: pd.DataFrame,
) -> pd.Series:
    """Compute market factor as equally-weighted average return.
    
    Args:
        all_returns: DataFrame of all stock returns
        
    Returns:
        Series of market returns (equal-weight)
    """
    market_factor = all_returns.mean(axis=1)
    return market_factor


def get_sector_exposures(
    weights: pd.Series,
    sector_mapping: Dict[str, str] = None,
) -> Dict[str, float]:
    """Compute sector exposures from portfolio weights.
    
    Args:
        weights: Series indexed by ticker with portfolio weights
        sector_mapping: Dict mapping ticker to sector
        
    Returns:
        Dict mapping sector to total weight
    """
    if sector_mapping is None:
        sector_mapping = SECTOR_MAPPING
    
    sector_exposures = {}
    for ticker, weight in weights.items():
        sector = sector_mapping.get(ticker, "Other")
        sector_exposures[sector] = sector_exposures.get(sector, 0.0) + weight
    
    return sector_exposures


def compute_factor_exposures(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    strategies_dict: Dict[str, Tuple[pd.Series, pd.DataFrame, dict]],
    rebalance_dates: List[pd.Timestamp],
    lookback_window_days: int = 252,
    window_beta: int = 60,
    sector_mapping: Dict[str, str] = None,
) -> pd.DataFrame:
    """Compute factor exposures for each strategy across backtest period.
    
    This function aggregates rolling factor exposures at each rebalance date
    and computes average factor loadings for each strategy.
    
    Args:
        prices: DataFrame of daily prices for all stocks
        returns: DataFrame of daily returns for all stocks
        strategies_dict: Dict mapping strategy name to (returns, weights_history, meta)
        rebalance_dates: List of rebalance dates from walk-forward backtest
        lookback_window_days: Lookback window for momentum/volatility
        window_beta: Rolling window for beta computation
        sector_mapping: Optional sector mapping dict
        
    Returns:
        DataFrame with factors as index, strategies as columns
    """
    if sector_mapping is None:
        sector_mapping = SECTOR_MAPPING
    
    # Compute market factor
    market_factor = compute_market_factor(returns)
    
    # Compute factors for each stock across all dates
    stock_betas = {}
    stock_momentum = {}
    stock_volatility = {}
    
    for ticker in returns.columns:
        stock_betas[ticker] = compute_rolling_beta(
            returns[ticker], 
            market_factor, 
            window=window_beta
        )
        stock_momentum[ticker] = compute_momentum_scores(prices[ticker])
        stock_volatility[ticker] = compute_volatility_scores(returns[ticker])
    
    # Aggregate by strategy and rebalance date
    results_by_strategy = {}
    
    for strategy_name, (strat_returns, weights_history, meta) in strategies_dict.items():
        
        # Collect exposures at each rebalance date
        beta_exposures = []
        momentum_exposures = []
        volatility_exposures = []
        
        for rebalance_date in rebalance_dates:
            if rebalance_date not in weights_history.index:
                continue
            
            weights = weights_history.loc[rebalance_date]
            weights = weights / weights.sum() if weights.sum() > 0 else weights
            
            # Exposure = sum(weight_i * factor_i)
            beta_exposure = 0.0
            momentum_exposure = 0.0
            volatility_exposure = 0.0
            
            for ticker in weights.index:
                w = weights[ticker]
                
                # Get closest valid factor values before rebalance date
                valid_betas = stock_betas[ticker][stock_betas[ticker].index <= rebalance_date]
                valid_momentum = stock_momentum[ticker][stock_momentum[ticker].index <= rebalance_date]
                valid_volatility = stock_volatility[ticker][stock_volatility[ticker].index <= rebalance_date]
                
                if len(valid_betas) > 0 and not np.isnan(valid_betas.iloc[-1]):
                    beta_exposure += w * valid_betas.iloc[-1]
                
                if len(valid_momentum) > 0 and not np.isnan(valid_momentum.iloc[-1]):
                    momentum_exposure += w * valid_momentum.iloc[-1]
                
                if len(valid_volatility) > 0 and not np.isnan(valid_volatility.iloc[-1]):
                    volatility_exposure += w * valid_volatility.iloc[-1]
            
            beta_exposures.append(beta_exposure)
            momentum_exposures.append(momentum_exposure)
            volatility_exposures.append(volatility_exposure)
        
        # Average exposures across rebalance dates
        results_by_strategy[strategy_name] = {
            'Beta': np.nanmean(beta_exposures) if beta_exposures else np.nan,
            'Momentum': np.nanmean(momentum_exposures) if momentum_exposures else np.nan,
            'Volatility': np.nanmean(volatility_exposures) if volatility_exposures else np.nan,
        }
        
        # Add latest sector exposures
        if len(weights_history) > 0:
            latest_weights = weights_history.iloc[-1]
            sector_exp = get_sector_exposures(latest_weights, sector_mapping)
            for sector, exp in sector_exp.items():
                results_by_strategy[strategy_name][f"Sector: {sector}"] = exp
    
    # Convert to DataFrame
    factor_exposure_df = pd.DataFrame(results_by_strategy).T
    
    # Reorder columns: main factors first, then sectors
    main_factors = ['Beta', 'Momentum', 'Volatility']
    sector_columns = [col for col in factor_exposure_df.columns if col.startswith('Sector:')]
    col_order = [col for col in main_factors if col in factor_exposure_df.columns]
    col_order.extend(sorted(sector_columns))
    
    factor_exposure_df = factor_exposure_df[col_order]
    
    return factor_exposure_df


def export_factor_exposures(
    factor_df: pd.DataFrame,
    output_path: str = "output/reports/factor_exposure_v2.csv",
) -> None:
    """Export factor exposures to CSV with formatting.
    
    Args:
        factor_df: DataFrame from compute_factor_exposures
        output_path: Path to save CSV file
    """
    # Round to 4 decimal places for readability
    factor_df_rounded = factor_df.round(4)
    factor_df_rounded.to_csv(output_path)
    print(f"✓ Factor exposures exported to {output_path}")


def print_factor_summary(factor_df: pd.DataFrame) -> None:
    """Print formatted summary of factor exposures.
    
    Args:
        factor_df: DataFrame from compute_factor_exposures
    """
    print("\n" + "="*80)
    print("FACTOR EXPOSURE ANALYSIS")
    print("="*80)
    print("\nInterpretation:")
    print("  Beta > 1.0     : Higher market sensitivity (amplified swings)")
    print("  Beta < 1.0     : Lower market sensitivity (more defensive)")
    print("  Momentum > 0   : Tilted toward trending stocks")
    print("  Volatility     : Average volatility of holdings")
    print("  Sector: *      : Portfolio weight in that sector")
    print("\n" + factor_df.to_string())
    print("\n" + "="*80 + "\n")
