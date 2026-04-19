"""Unit tests for stress testing module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.stress import (
    define_historical_scenarios,
    compute_stress_metrics,
    run_stress_analysis,
    stress_summary_table,
    export_stress_results,
    print_stress_summary,
)


class TestDefineScenarios:
    """Test scenario definition functions."""
    
    def test_historical_scenarios_structure(self):
        """Verify historical scenarios return correct structure."""
        scenarios = define_historical_scenarios()
        
        assert isinstance(scenarios, dict)
        assert len(scenarios) == 3
        
        for name, (start_date, end_date) in scenarios.items():
            assert isinstance(name, str)
            assert isinstance(start_date, str)
            assert isinstance(end_date, str)
            # Verify date format (YYYY-MM-DD)
            pd.to_datetime(start_date)
            pd.to_datetime(end_date)
            # Verify start <= end
            assert pd.to_datetime(start_date) <= pd.to_datetime(end_date)
    
    def test_historical_scenarios_have_valid_date_ranges(self):
        """Verify all historical scenario start dates precede end dates."""
        scenarios = define_historical_scenarios()
        for name, (start, end) in scenarios.items():
            assert pd.to_datetime(start) <= pd.to_datetime(end), f"Invalid range in {name}"


class TestComputeStressMetrics:
    """Test stress metrics computation."""
    
    @pytest.fixture
    def synthetic_returns(self):
        """Create synthetic return series for testing."""
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        
        # Strategy 1: steady 10% annual return
        strategy1_returns = np.random.normal(0.0004, 0.01, 252)  # ~10% ann, ~1% vol
        strategy1 = pd.Series(strategy1_returns, index=dates, name="steady")
        
        # Strategy 2: includes large drawdown
        strategy2_returns = np.random.normal(0.0004, 0.01, 252)
        # Add large shock in middle period
        strategy2_returns[80:100] = -0.05  # Large negative shock
        strategy2 = pd.Series(strategy2_returns, index=dates, name="volatile")
        
        return {
            "Steady": strategy1,
            "Volatile": strategy2,
        }
    
    def test_compute_stress_metrics_output_shape(self, synthetic_returns):
        """Verify output has correct shape and structure."""
        scenarios = {
            "Scenario 1": ("2020-01-01", "2020-06-30"),
            "Scenario 2": ("2020-07-01", "2020-12-31"),
        }
        
        result = compute_stress_metrics(synthetic_returns, scenarios)
        
        # Check multi-index structure
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["Scenario", "Strategy"]
        
        # Check expected rows (2 scenarios × 2 strategies)
        assert len(result) == 4
        
        # Check expected columns
        expected_columns = {"Cumulative Return", "Annualized Vol", "Max Drawdown", "Calmar Ratio", "N Days", "Note"}
        assert expected_columns == set(result.columns)
    
    def test_compute_stress_metrics_values(self, synthetic_returns):
        """Verify computed metrics are reasonable."""
        scenarios = {"Test": ("2020-01-01", "2020-12-31")}
        result = compute_stress_metrics(synthetic_returns, scenarios)
        
        # Cumulative return should be between -1 and 1 (reasonable for annual period)
        cum_ret = result["Cumulative Return"].values
        assert np.all((cum_ret >= -1) & (cum_ret <= 1) | np.isnan(cum_ret))
        
        # Max drawdown should be <= 0
        max_dd = result["Max Drawdown"].values
        assert np.all((max_dd <= 0) | np.isnan(max_dd))
        
        # Annualized vol should be positive
        ann_vol = result["Annualized Vol"].values
        assert np.all((ann_vol >= 0) | np.isnan(ann_vol))
    
    def test_edge_case_no_data_in_window(self, synthetic_returns):
        """Verify graceful handling when scenario window has no data."""
        # Create scenario with no overlapping dates
        scenarios = {"Future": ("2030-01-01", "2030-12-31")}
        
        result = compute_stress_metrics(synthetic_returns, scenarios)
        
        # Should have NaN values
        assert result["Cumulative Return"].isna().all()
        assert result["Max Drawdown"].isna().all()
        assert result["Annualized Vol"].isna().all()
        assert result["Calmar Ratio"].isna().all()
    
    def test_edge_case_single_day_scenario(self, synthetic_returns):
        """Verify handling of single-day scenario."""
        scenarios = {"Single Day": ("2020-01-02", "2020-01-02")}
        
        result = compute_stress_metrics(synthetic_returns, scenarios)
        
        # Should still compute (though potentially meaningless)
        assert not result.empty
        # Single-day return should be available
        assert not np.isnan(result["Cumulative Return"].iloc[0])
    
    def test_date_filtering_accuracy(self, synthetic_returns):
        """Verify correct date masking for scenario windows."""
        dates = synthetic_returns["Steady"].index
        start_idx = 30
        end_idx = 60
        
        start_date = dates[start_idx].strftime("%Y-%m-%d")
        end_date = dates[end_idx].strftime("%Y-%m-%d")
        
        scenarios = {"Test Window": (start_date, end_date)}
        result = compute_stress_metrics(synthetic_returns, scenarios)
        
        # Manually verify the window
        mask = (dates >= start_date) & (dates <= end_date)
        manual_returns = synthetic_returns["Steady"][mask]
        manual_cumul_return = (1 + manual_returns).prod() - 1
        
        computed_cumul_return = result.loc[("Test Window", "Steady"), "Cumulative Return"]
        
        assert np.isclose(manual_cumul_return, computed_cumul_return)


class TestStressSummary:
    """Test summary table generation."""
    
    @pytest.fixture
    def sample_stress_df(self):
        """Create sample stress DataFrame."""
        index = pd.MultiIndex.from_product(
            [["Scenario A", "Scenario B"], ["Strategy X", "Strategy Y"]],
            names=["Scenario", "Strategy"]
        )
        
        data = {
            "Cumulative Return": [0.05, -0.10, 0.02, -0.15],
            "Annualized Vol": [0.10, 0.12, 0.09, 0.14],
            "Max Drawdown": [-0.05, -0.15, -0.03, -0.20],
            "Calmar Ratio": [1.0, -0.67, 0.67, -0.75],
            "N Days": [126, 126, 126, 126],
            "Note": ["", "", "", ""],
        }
        
        return pd.DataFrame(data, index=index)
    
    def test_stress_summary_pivot_structure(self, sample_stress_df):
        """Verify pivoted summary has correct structure."""
        summary = stress_summary_table(sample_stress_df)
        
        # Should be a multi-column index (metric, strategy)
        assert isinstance(summary.columns, pd.MultiIndex)
        
        # Should have 2 rows (scenarios)
        assert len(summary) == 2


class TestExportResults:
    """Test result export functionality."""
    
    @pytest.fixture
    def sample_stress_df(self):
        """Create sample stress DataFrame."""
        index = pd.MultiIndex.from_product(
            [["Scenario A"], ["Strategy X"]],
            names=["Scenario", "Strategy"]
        )
        
        data = {
            "Cumulative Return": [0.05],
            "Annualized Vol": [0.10],
            "Max Drawdown": [-0.05],
            "Calmar Ratio": [1.0],
            "N Days": [126],
            "Note": [""],
        }
        
        return pd.DataFrame(data, index=index)
    
    def test_export_creates_files(self, sample_stress_df, tmp_path):
        """Verify export creates expected CSV files."""
        output_dir = tmp_path / "reports"
        
        export_stress_results(sample_stress_df, output_dir, version="test")
        
        # Check files exist
        assert (output_dir / "stress_metrics_test.csv").exists()
        assert (output_dir / "stress_summary_test.csv").exists()
    
    def test_export_file_content(self, sample_stress_df, tmp_path):
        """Verify exported files have correct content."""
        output_dir = tmp_path / "reports"
        
        export_stress_results(sample_stress_df, output_dir, version="test")
        
        # Read back and verify
        metrics_df = pd.read_csv(
            output_dir / "stress_metrics_test.csv",
            index_col=["Scenario", "Strategy"]
        )
        
        assert not metrics_df.empty
        assert "Cumulative Return" in metrics_df.columns
        assert "Max Drawdown" in metrics_df.columns


class TestRunStressAnalysis:
    """Test orchestrator function."""
    
    @pytest.fixture
    def synthetic_returns(self):
        """Create synthetic return series."""
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        returns = np.random.normal(0.0004, 0.01, 252)
        
        return {
            "Strategy A": pd.Series(returns, index=dates),
            "Strategy B": pd.Series(returns * 0.9, index=dates),
        }
    
    def test_run_stress_analysis_output(self, synthetic_returns):
        """Verify orchestrator returns correct format."""
        scenarios = {"Test": ("2020-01-01", "2020-12-31")}
        
        result = run_stress_analysis(synthetic_returns, scenarios)
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert len(result) == 2  # 1 scenario × 2 strategies


class TestValidation:
    """Integration and validation tests."""
    
    def test_full_pipeline(self):
        """Test full stress testing pipeline."""
        # Create realistic test data
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        
        returns_dict = {
            "Strategy 1": pd.Series(np.random.normal(0.0005, 0.015, 500), index=dates),
            "Strategy 2": pd.Series(np.random.normal(0.0004, 0.012, 500), index=dates),
        }
        
        scenarios = {
            "Period 1": ("2020-01-01", "2020-06-30"),
            "Period 2": ("2020-07-01", "2020-12-31"),
        }
        
        # Run analysis
        result = run_stress_analysis(returns_dict, scenarios)
        
        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 2 scenarios × 2 strategies
        assert not result.isna().all().all(), "Should have some non-NaN values"
    
    def test_historical_scenarios_valid(self):
        """Verify historical scenarios use valid dates."""
        scenarios = define_historical_scenarios()
        
        for name, (start, end) in scenarios.items():
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            assert start_dt <= end_dt, f"Invalid dates in {name}"
            # Verify reasonable date ranges (2020 onwards)
            assert start_dt.year >= 2020, f"Start date too early in {name}"
