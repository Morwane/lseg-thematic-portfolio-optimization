"""Data fetching module for LSEG thematic portfolio optimization."""

from dataclasses import dataclass
from typing import List

import pandas as pd

try:
    import lseg.data as ld
except ImportError as e:
    raise ImportError("lseg.data is not installed. Please install lseg-data package.") from e


@dataclass
class MarketDataRequest:
    """Request object for market data fetching."""

    tickers: List[str]
    start_date: str
    end_date: str


def fetch_prices_lseg(request: MarketDataRequest) -> pd.DataFrame:
    """Fetch daily prices from LSEG Data API using TRDPRC_1 field.

    Args:
        request: MarketDataRequest with tickers, start_date, end_date

    Returns:
        DataFrame indexed by DatetimeIndex, columns are ticker strings

    Raises:
        ValueError: If fetched data is empty
        ImportError: If lseg.data is not installed
    """
    session = None
    try:
        session = ld.open_session()

        prices = ld.get_history(
            universe=request.tickers,
            fields=["TRDPRC_1"],
            start=request.start_date,
            end=request.end_date,
            interval="daily",
        )

        if prices is None or prices.empty:
            raise ValueError(
                f"No data returned from LSEG for tickers {request.tickers} "
                f"between {request.start_date} and {request.end_date}"
            )

        # Ensure index is DatetimeIndex
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)

        # Extract price column and reshape for multiple tickers
        if "TRDPRC_1" in prices.columns:
            prices = prices["TRDPRC_1"]

        # Handle single ticker case (returns Series)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=request.tickers[0])

        # Ensure columns are ticker strings
        prices.columns = request.tickers

        # Convert to numeric, coercing errors to NaN
        prices = prices.astype(float)

        return prices

    finally:
        if session is not None:
            session.close()
