"""Tests for execution pipeline — signal generation and trade management."""

import numpy as np
import pytest

from scalp2.execution.trade_manager import TradeManager, TradeState, TradeStatus
from scalp2.config import TradeManagementConfig


@pytest.fixture
def trade_mgr():
    config = TradeManagementConfig()
    return TradeManager(config, max_holding_bars=10)


@pytest.fixture
def long_trade():
    return TradeState(
        direction="LONG",
        entry_price=50000.0,
        current_stop_loss=49900.0,
        take_profit=50120.0,
        atr_at_entry=100.0,
    )


class TestTradeManager:
    def test_stop_loss_closes_trade(self, trade_mgr, long_trade):
        # Price drops to SL
        trade_mgr.update(long_trade, current_high=50050, current_low=49850, current_close=49860)
        assert long_trade.status == TradeStatus.CLOSED_SL

    def test_partial_tp(self, trade_mgr, long_trade):
        # Price moves 0.6 ATR above entry (50060)
        trade_mgr.update(long_trade, current_high=50070, current_low=50010, current_close=50065)
        assert long_trade.status == TradeStatus.PARTIAL_TP
        assert long_trade.remaining_size < 1.0
        # SL should be at breakeven
        assert long_trade.current_stop_loss == long_trade.entry_price

    def test_time_barrier(self, trade_mgr, long_trade):
        # Simulate max_holding bars with no barrier hit
        for _ in range(10):
            trade_mgr.update(long_trade, current_high=50030, current_low=49920, current_close=50010)
            if long_trade.status != TradeStatus.OPEN:
                break
        assert long_trade.status == TradeStatus.CLOSED_TIME

    def test_regime_change_closes(self, trade_mgr, long_trade):
        trade_mgr.update(
            long_trade, current_high=50050, current_low=49950,
            current_close=50010, is_choppy=True
        )
        assert long_trade.status == TradeStatus.CLOSED_REGIME

    def test_pnl_positive_on_tp(self, trade_mgr):
        trade = TradeState(
            direction="LONG",
            entry_price=50000.0,
            current_stop_loss=49900.0,
            take_profit=50120.0,
            atr_at_entry=100.0,
        )
        # Hit full TP (1.2 ATR = 120 points above entry)
        trade_mgr.update(trade, current_high=50200, current_low=50050, current_close=50150)
        assert trade.pnl > 0
