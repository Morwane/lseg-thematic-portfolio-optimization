"""Data fetching module for LSEG thematic portfolio optimization."""

from dataclasses import dataclass
import time
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


def _open_lseg_session():
    """Open an LSEG session with a couple of known desktop aliases."""
    errors = []

    # Try explicit desktop session first, then fallback to default.
    for session_name in ("desktop.workspace", "default"):
        try:
            if session_name == "default":
                return ld.open_session(), session_name
            return ld.open_session(session_name), session_name
        except Exception as exc:  # pragma: no cover - depends on local LSEG runtime
            errors.append(f"{session_name}: {exc}")

    raise ConnectionError(
        "Unable to open LSEG session. "
        "Ensure LSEG Workspace/Desktop is running and API Proxy is enabled. "
        f"Details: {' | '.join(errors)}"
    )


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
    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        session = None
        try:
            session, session_name = _open_lseg_session()

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

        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                # Desktop proxy can flap briefly; retry with short backoff.
                time.sleep(1.5 * attempt)
            continue

        finally:
            try:
                if session is not None and hasattr(session, "close"):
                    session.close()
                elif hasattr(ld, "close_session"):
                    ld.close_session()
            except Exception:
                pass

    raise ConnectionError(
        "LSEG session is unavailable after retries. "
        "If Workspace/Desktop is open, verify API Proxy connectivity and try again. "
        f"Last error: {last_error}"
    )
