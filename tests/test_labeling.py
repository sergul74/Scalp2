"""Tests for triple barrier labeling."""

import numpy as np
import pandas as pd
import pytest

from scalp2.config import LabelConfig
from scalp2.labeling.triple_barrier import triple_barrier_labels


@pytest.fixture
def trending_up_df():
    """Create data with a clear uptrend — should produce Long labels."""
    n = 200
    close = 50000 + np.arange(n) * 50.0  # Steady uptrend
    high = close + 30
    low = close - 20
    atr = np.full(n, 100.0)  # Constant ATR of 100

    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": close - 10,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 1000,
            "atr_14": atr,
        },
        index=idx,
        dtype=np.float32,
    )


@pytest.fixture
def trending_down_df():
    """Create data with a clear downtrend — should produce Short labels."""
    n = 200
    close = 50000 - np.arange(n) * 50.0
    high = close + 20
    low = close - 30
    atr = np.full(n, 100.0)

    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": close + 10,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 1000,
            "atr_14": atr,
        },
        index=idx,
        dtype=np.float32,
    )


class TestTripleBarrier:
    def test_uptrend_produces_long_labels(self, trending_up_df):
        config = LabelConfig(tp_multiplier=1.2, sl_multiplier=1.0, max_holding_bars=10)
        result = trending_up_df.copy()
        # Need to ensure high values break the TP barrier
        result["high"] = result["close"] + 200  # Ensure TP hit
        labeled = triple_barrier_labels(result, config)
        long_ratio = (labeled["tb_label"] == 1).mean()
        assert long_ratio > 0.5, f"Expected majority Long labels in uptrend, got {long_ratio:.2f}"

    def test_downtrend_produces_short_labels(self, trending_down_df):
        config = LabelConfig(tp_multiplier=1.2, sl_multiplier=1.0, max_holding_bars=10)
        result = trending_down_df.copy()
        result["low"] = result["close"] - 200  # Ensure SL hit for longs
        labeled = triple_barrier_labels(result, config)
        short_ratio = (labeled["tb_label"] == -1).mean()
        assert short_ratio > 0.3, f"Expected significant Short labels in downtrend, got {short_ratio:.2f}"

    def test_label_values(self, trending_up_df):
        config = LabelConfig()
        labeled = triple_barrier_labels(trending_up_df, config)
        valid_labels = {-1, 0, 1}
        actual_labels = set(labeled["tb_label"].unique())
        assert actual_labels.issubset(valid_labels), f"Invalid labels: {actual_labels}"

    def test_classifier_label_remap(self, trending_up_df):
        config = LabelConfig()
        labeled = triple_barrier_labels(trending_up_df, config)
        valid_cls = {0, 1, 2}
        actual_cls = set(labeled["tb_label_cls"].unique())
        assert actual_cls.issubset(valid_cls), f"Invalid classifier labels: {actual_cls}"

    def test_no_sentinel_values(self, trending_up_df):
        config = LabelConfig()
        labeled = triple_barrier_labels(trending_up_df, config)
        assert -999 not in labeled["tb_label"].values
        assert -999 not in labeled["tb_label_cls"].values

    def test_return_column_exists(self, trending_up_df):
        config = LabelConfig()
        labeled = triple_barrier_labels(trending_up_df, config)
        assert "tb_return" in labeled.columns
        assert labeled["tb_return"].dtype == np.float32
