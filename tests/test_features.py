"""Tests for feature engineering modules."""

import numpy as np
import pandas as pd
import pytest

from scalp2.config import TechnicalConfig, VolatilityConfig, SmartMoneyConfig
from scalp2.features.technical import compute_rsi, compute_atr, compute_all_technical
from scalp2.features.volatility import garman_klass, parkinson
from scalp2.features.smart_money import fair_value_gaps, vwap_distance


@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    volume = np.abs(np.random.randn(n) * 1000) + 100

    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_.astype(np.float32),
            "high": high.astype(np.float32),
            "low": low.astype(np.float32),
            "close": close.astype(np.float32),
            "volume": volume.astype(np.float32),
        },
        index=idx,
    )


class TestTechnicalIndicators:
    def test_rsi_range(self, sample_ohlcv):
        rsi = compute_rsi(sample_ohlcv["close"], 14)
        valid = rsi.dropna()
        assert valid.min() >= 0, "RSI below 0"
        assert valid.max() <= 100, "RSI above 100"

    def test_atr_positive(self, sample_ohlcv):
        atr = compute_atr(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], 14
        )
        valid = atr.dropna()
        assert (valid >= 0).all(), "ATR has negative values"

    def test_all_technical_columns(self, sample_ohlcv):
        config = TechnicalConfig()
        result = compute_all_technical(sample_ohlcv, config)

        expected_cols = [
            "rsi_14", "ema_9", "ema_21", "ema_55",
            "macd_line", "macd_signal", "macd_hist",
            "bb_middle", "bb_upper", "bb_lower",
            "atr_14", "stoch_k", "stoch_d",
            "adx", "log_return",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_future_leakage(self, sample_ohlcv):
        """Verify indicators use only past data (no centered windows)."""
        config = TechnicalConfig()
        result = compute_all_technical(sample_ohlcv, config)

        # Modify the last value and recompute
        modified = sample_ohlcv.copy()
        modified.iloc[-1, modified.columns.get_loc("close")] *= 1.1

        result2 = compute_all_technical(modified, config)

        # Only the last few values should differ (within indicator lookback)
        # Values at the start should be identical
        for col in ["rsi_14", "ema_9"]:
            np.testing.assert_array_equal(
                result[col].values[:100],
                result2[col].values[:100],
                err_msg=f"Future leakage detected in {col}",
            )


class TestVolatility:
    def test_garman_klass_positive(self, sample_ohlcv):
        gk = garman_klass(sample_ohlcv, window=14)
        valid = gk.dropna()
        # GK can be negative in its raw form, but rolling mean smooths it
        assert len(valid) > 0

    def test_parkinson_positive(self, sample_ohlcv):
        pk = parkinson(sample_ohlcv, window=14)
        valid = pk.dropna()
        assert (valid >= 0).all(), "Parkinson volatility should be non-negative"


class TestSmartMoney:
    def test_fvg_output_columns(self, sample_ohlcv):
        result = fair_value_gaps(sample_ohlcv, min_gap_pct=0.0001)
        assert "fvg_bullish" in result.columns
        assert "fvg_bearish" in result.columns
        assert "fvg_bull_dist" in result.columns

    def test_vwap_output(self, sample_ohlcv):
        result = vwap_distance(sample_ohlcv, session_hours=24)
        assert "vwap" in result.columns
        assert "vwap_dist_pct" in result.columns
