"""Backtest Optimizer — find optimal ATR/Confidence thresholds.

This script parses raw local data, builds multi-timeframe features,
runs model inference on all historical bars, and simulates trading 
using exact TradeManager logic for different ATR thresholds.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scalp2.config import load_config
from scalp2.utils.serialization import load_fold_artifacts
from scalp2.data.preprocessing import clean_ohlcv, resample_ohlcv, load_binance_csv
from scalp2.features.builder import build_features, drop_warmup_nans
from scalp2.data.mtf_builder import build_mtf_dataset
from scalp2.data.dataset import ScalpDataset
from scalp2.models.hybrid import HybridEncoder
from scalp2.models.meta_learner import XGBoostMetaLearner
from scalp2.execution.trade_manager import TradeManager, TradeState, TradeStatus

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


def prepare_data(df_raw: pd.DataFrame, config) -> pd.DataFrame:
    """Clean, resample, and build MTF features matching the notebook."""
    print("Building MTF features... (this might take a minute)")
    
    df_15m = clean_ohlcv(df_raw, "15m")
    df_1h = resample_ohlcv(df_15m, "1h")
    df_4h = resample_ohlcv(df_15m, "4h")

    df_15m_feat = build_features(df_15m, config.features)
    df_1h_feat = build_features(df_1h, config.features)
    df_4h_feat = build_features(df_4h, config.features)

    df_full = build_mtf_dataset(df_15m_feat, df_1h_feat, df_4h_feat)
    df_full = drop_warmup_nans(df_full)
    
    # Calculate ATR percentile for the filter
    if "atr_14" in df_full.columns:
        df_full["atr_pctile"] = df_full["atr_14"].rolling(96, min_periods=10).rank(pct=True).fillna(1.0)
    else:
        df_full["atr_pctile"] = 1.0

    print(f"Data ready: {len(df_full)} bars")
    return df_full


def run_inference(df_full: pd.DataFrame, config, artifacts, device) -> tuple[np.ndarray, np.ndarray]:
    """Run HybridEncoder + XGBoost on all bars to get probabilities and regime."""
    print("Running deep learning inference on all bars...")
    seq_len = config.model.seq_len
    
    feature_names = artifacts["feature_names"]
    scaler = artifacts["scaler"]
    
    # Missing columns handling
    for col in feature_names:
        if col not in df_full.columns:
            df_full[col] = 0.0
            
    raw_X = df_full[feature_names].values.astype(np.float32)
    scaled_X = scaler.transform(raw_X).astype(np.float32)
    scaled_X = np.nan_to_num(scaled_X, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Model init
    encoder = HybridEncoder(len(feature_names), config.model).to(device)
    encoder.load_state_dict(artifacts["model_state"])
    encoder.eval()
    
    # 2. Extract latents
    dummy_l = np.zeros(len(scaled_X), dtype=np.int64)
    dummy_r = np.zeros(len(scaled_X), dtype=np.float32)
    ds = ScalpDataset(scaled_X, dummy_l, dummy_r, seq_len)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    
    latents = []
    with torch.no_grad():
        for bx, _, _ in loader:
            _, lat = encoder(bx.to(device))
            latents.append(lat.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    
    # 3. Regime
    df_aligned = df_full.iloc[seq_len:]
    regime_detector = artifacts.get("regime_detector", None)
    if regime_detector:
        regime_probs = regime_detector.predict_proba_online(df_aligned)
    else:
        regime_probs = np.full((len(df_aligned), 3), 1/3, dtype=np.float32)
        
    # 4. Meta features & XGBoost
    top_indices = artifacts["top_feature_indices"]
    hc = scaled_X[seq_len:][:, top_indices]
    
    meta_X = XGBoostMetaLearner.build_meta_features(latents, hc, regime_probs)
    xgb = XGBoostMetaLearner(config.model.xgboost)
    xgb_path = Path(artifacts["checkpoint_dir"]) / f"xgb_fold_{artifacts['fold_idx']:03d}.json"
    xgb.load(str(xgb_path))
    
    probs = xgb.predict_proba(meta_X)
    print("Inference complete.")
    return probs, regime_probs


def simulate_trading(
    df: pd.DataFrame, 
    probs: np.ndarray, 
    regime_probs: np.ndarray, 
    config, 
    min_atr_percentile: float,
    confidence_threshold: float,
) -> dict:
    """Run walk-forward trade simulation using TradeManager."""
    exec_cfg = config.execution
    label_cfg = config.labeling
    trade_mgmt_cfg = config.execution.trade_management
    
    trade_mgr = TradeManager(trade_mgmt_cfg, label_cfg.max_holding_bars)
    
    active = None
    pending = None
    daily_count = 0
    prev_date = None
    
    seq_len = config.model.seq_len
    n_bars = len(probs)
    
    # Kelly setup
    effective_tp = (trade_mgmt_cfg.partial_tp_1_pct * trade_mgmt_cfg.partial_tp_1_atr + 
                    (1 - trade_mgmt_cfg.partial_tp_1_pct) * trade_mgmt_cfg.full_tp_atr)
    kelly_b = effective_tp / label_cfg.sl_multiplier
    kelly_fraction = exec_cfg.position_sizing.kelly_fraction
    kelly_max = exec_cfg.position_sizing.max_fraction
    
    n_trades = 0
    n_wins = 0
    n_losses = 0
    gross_pnl_pct = 0.0
    skip_atr = 0
    skip_conf = 0
    skip_choppy = 0
    skip_adx = 0
    
    for i in range(n_bars):
        real_idx = seq_len + i
        if real_idx >= len(df): break
            
        row = df.iloc[real_idx]
        cur_date = row.name.date() if hasattr(row.name, 'date') else None
        if cur_date != prev_date:
            daily_count = 0
            prev_date = cur_date

        # Update active trade
        if active is not None:
            is_choppy = regime_probs[i, 2] > config.regime.choppy_threshold if i < len(regime_probs) else False
            active = trade_mgr.update(active, row['high'], row['low'], row['close'], is_choppy)
            
            if active.status not in (TradeStatus.OPEN, TradeStatus.PARTIAL_TP):
                n_trades += 1
                if active.pnl > 0:
                    n_wins += 1
                else:
                    n_losses += 1
                # Simplified PnL for analysis (unleveraged gross)
                gross_pnl_pct += float(active.pnl)
                active = None
            continue
            
        # Execute pending signal
        if pending is not None:
            ps = pending
            pending = None
            
            # Slippage/costs omitted in loop for speed, we calculate gross edge here
            sl = ps["entry"] - label_cfg.sl_multiplier * ps["atr"] if ps["dir"] == "LONG" else ps["entry"] + label_cfg.sl_multiplier * ps["atr"]
            
            active = TradeState(
                direction=ps["dir"],
                entry_price=ps["entry"],
                current_stop_loss=sl,
                take_profit=0.0,
                atr_at_entry=ps["atr"]
            )
            daily_count += 1
            continue

        # Look for new signals
        p = probs[i]
        cls = int(np.argmax(p))
        if cls == 1: continue  # Hold

        conf = max(p[0], p[2])
        if conf < confidence_threshold:
            skip_conf += 1
            continue
            
        if daily_count >= exec_cfg.max_trades_per_day:
            continue
            
        if exec_cfg.time_of_day_filter.enabled:
            current_time = row.name
            hour = current_time.hour if hasattr(current_time, "hour") else pd.to_datetime(current_time).hour
            if hour in exec_cfg.time_of_day_filter.blocked_hours_utc:
                continue
            
        atr = row.get("atr_14", 0.0)
        atr_pct = row.get("atr_pctile", 1.0)
        adx = row.get("adx_14", 999.0)
        
        if adx < exec_cfg.min_adx:
            skip_adx += 1
            continue
            
        if atr_pct < min_atr_percentile:
            skip_atr += 1
            continue
            
        if regime_probs[i, 2] > config.regime.choppy_threshold:
            skip_choppy += 1
            continue

        if real_idx + 1 >= len(df): continue
            
        pending = {
            "dir": "LONG" if cls == 2 else "SHORT",
            "entry": df.iloc[real_idx + 1]["open"],
            "atr": atr,
            "confidence": conf
        }

    return {
        "trades": n_trades, "wins": n_wins, "losses": n_losses, "gross_pnl_pct": gross_pnl_pct,
        "skipped_atr": skip_atr, "skipped_conf": skip_conf, "skipped_choppy": skip_choppy, "skipped_adx": skip_adx
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load artifacts
    checkpoint_dir = Path(args.checkpoint_dir)
    fold_dirs = sorted([d.name for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    fold_idx = int(fold_dirs[-1].split("_")[1])
    artifacts = load_fold_artifacts(checkpoint_dir, fold_idx, device=device)
    artifacts["checkpoint_dir"] = checkpoint_dir
    artifacts["fold_idx"] = fold_idx

    # 2. Data
    df_raw = load_binance_csv("data/btcusdt_15min.csv")

    df_full = prepare_data(df_raw, config)
    
    # 3. Predict all
    probs, regime_probs = run_inference(df_full, config, artifacts, device)

    # 4. ATR Sweep
    print("\n" + "="*60)
    print("  SCALP2 OPTIMIZATION — ATR PERCENTILE SWEEP")
    print("="*60)
    print(f"| {'ATR Pct':<8} | {'Trades':<8} | {'Gross PnL':<10} | {'Win Rate':<10} |")
    print("-" * 60)
    
    best_pnl = -999.0
    best_atr = 0.0
    
    for atr_thresh in np.arange(0.0, 0.35, 0.05):
        stats = simulate_trading(df_full, probs, regime_probs, config, atr_thresh, config.execution.confidence_threshold)
        
        wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
        pnl = stats["gross_pnl_pct"] * 100  # in percent unleveraged
        
        print(f"| {atr_thresh:<8.2f} | {stats['trades']:<8d} | {pnl:+.2f}%     | {wr:<6.1f}%   |")
        
        if pnl > best_pnl and stats["trades"] > 10:
            best_pnl = pnl
            best_atr = atr_thresh

    print("-" * 60)
    print(f"💡 Recommendation: min_atr_percentile = {best_atr:.2f} (yields {best_pnl:+.2f}% gross)")
    
    # 5. Confidence Sweep
    print("\n" + "="*60)
    print(f"  SCALP2 OPTIMIZATION — CONFIDENCE SWEEP (ATR={best_atr:.2f})")
    print("="*60)
    print(f"| {'Conf':<8} | {'Trades':<8} | {'Gross PnL':<10} | {'Win Rate':<10} |")
    print("-" * 60)

    for conf_thresh in np.arange(0.40, 0.70, 0.05):
        stats = simulate_trading(df_full, probs, regime_probs, config, best_atr, conf_thresh)
        wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
        pnl = stats["gross_pnl_pct"] * 100
        print(f"| {conf_thresh:<8.2f} | {stats['trades']:<8d} | {pnl:+.2f}%     | {wr:<6.1f}%   |")
        
    print("="*60)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
