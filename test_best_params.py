import pandas as pd
import numpy as np
from numba import njit, prange
import os
import glob
from datetime import datetime, timedelta
import time
import gc

# ==========================================
# VALIDATION TEST CONFIGURATION
# ==========================================
DATA_PATH = r"d:\repos\ethusd\data\raw"
RESULTS_FILE = "optimization_results.csv"

# You can change these dates to test the top 20 on a NEW date range 
# (Out-of-Sample testing) or the same date range to verify.
START_DATE = "2026-02-01"
END_DATE = "2026-02-28"

INITIAL_BALANCE = 100.0
LEVERAGE = 400
SPREAD_PIPS = 0.5
WIB_OFFSET = 7
# ==========================================

@njit
def simulate_ticks(ticks, buy_stop, sell_stop, lot_size_in, tp_distance, sl_price_buy, sl_price_sell, spread, current_balance, leverage):
    """Core tick simulator"""
    pos_type = 0 
    entry_price, tp_price, sl_price = 0.0, 0.0, 0.0
    lot_size = lot_size_in
    trades_count, total_profit = 0, 0.0
    t1_entry, t1_exit, t1_p, t1_mdd, t1_margin = -1, -1, 0.0, 0.0, 0.0
    t2_entry, t2_exit, t2_p, t2_mdd, t2_margin = -1, -1, 0.0, 0.0, 0.0
    buy_pending, sell_pending = True, True
    active_balance = current_balance
    s2 = spread / 2.0
    
    for i in range(len(ticks)):
        p = ticks[i]
        if pos_type == 0:
            if buy_pending and p >= buy_stop:
                entry_price = buy_stop + s2
                t1_margin = (entry_price * lot_size) / leverage
                if t1_margin > active_balance: return 0.0, 1, -2, t1_mdd, t2_mdd 
                pos_type = 1
                tp_price = buy_stop + tp_distance 
                sl_price = sl_price_buy - s2
                trades_count = 1
                buy_pending = False
            elif sell_pending and p <= sell_stop:
                entry_price = sell_stop - s2
                t1_margin = (entry_price * lot_size) / leverage
                if t1_margin > active_balance: return 0.0, 1, -2, t1_mdd, t2_mdd
                pos_type = 2
                tp_price = sell_stop - tp_distance
                sl_price = sl_price_sell + s2
                trades_count = 1
                sell_pending = False
                
        elif pos_type == 1: 
            unrealized_p = (p - entry_price) * lot_size
            floating_equity = active_balance - ((entry_price * lot_size) / leverage) + unrealized_p
            drawdown_dollars = active_balance - floating_equity
            
            if drawdown_dollars > 0:
                if trades_count == 1: t1_mdd = max(t1_mdd, drawdown_dollars)
                else: t2_mdd = max(t2_mdd, drawdown_dollars)
                
            if floating_equity <= 0:
                if trades_count == 1: return unrealized_p, 1, -2, t1_mdd, t2_mdd
                else: return t1_p + unrealized_p, 2, -2, t1_mdd, t2_mdd
                
            if p >= (tp_price + s2): 
                res_p = (tp_price - entry_price) * lot_size
                if trades_count == 1: return res_p, 1, 1, t1_mdd, t2_mdd
                else: return t1_p + res_p, 2, 1, t1_mdd, t2_mdd
            elif p <= sl_price: 
                res_p = (sl_price - entry_price) * lot_size
                if trades_count == 1:
                    t1_p = res_p
                    active_balance += res_p 
                    new_entry_price = sl_price - s2
                    new_lot_size = lot_size * 2.0
                    t2_margin = (new_entry_price * new_lot_size) / leverage
                    if t2_margin > active_balance: return t1_p, 2, -2, t1_mdd, t2_mdd
                    pos_type, entry_price, lot_size = 2, new_entry_price, new_lot_size
                    tp_price, sl_price = sl_price - tp_distance, buy_stop + s2
                    trades_count = 2
                else:
                    return t1_p + res_p, 2, -1, t1_mdd, t2_mdd
                    
        elif pos_type == 2: 
            unrealized_p = (entry_price - p) * lot_size
            floating_equity = active_balance - ((entry_price * lot_size) / leverage) + unrealized_p
            drawdown_dollars = active_balance - floating_equity
            
            if drawdown_dollars > 0:
                if trades_count == 1: t1_mdd = max(t1_mdd, drawdown_dollars)
                else: t2_mdd = max(t2_mdd, drawdown_dollars)

            if floating_equity <= 0:
                if trades_count == 1: return unrealized_p, 1, -2, t1_mdd, t2_mdd
                else: return t1_p + unrealized_p, 2, -2, t1_mdd, t2_mdd

            if p <= (tp_price - s2): 
                res_p = (entry_price - tp_price) * lot_size
                if trades_count == 1: return res_p, 1, 1, t1_mdd, t2_mdd
                else: return t1_p + res_p, 2, 1, t1_mdd, t2_mdd
            elif p >= sl_price: 
                res_p = (entry_price - sl_price) * lot_size
                if trades_count == 1:
                    t1_p = res_p
                    active_balance += res_p 
                    new_entry_price = sl_price + s2
                    new_lot_size = lot_size * 2.0
                    t2_margin = (new_entry_price * new_lot_size) / leverage
                    if t2_margin > active_balance: return t1_p, 2, -2, t1_mdd, t2_mdd
                    pos_type, entry_price, lot_size = 1, new_entry_price, new_lot_size
                    tp_price, sl_price = sl_price + tp_distance, sell_stop - s2
                    trades_count = 2
                else:
                    return t1_p + res_p, 2, -1, t1_mdd, t2_mdd
                    
    return total_profit, trades_count, 0, t1_mdd, t2_mdd

@njit(parallel=True)
def run_fast_grid_search_chunk(param_matrix, state_matrix, prices, timestamps, chunk_start_ts_ns, days_in_chunk, leverage, spread):
    """Executes the grid search iteratively across all CPU cores."""
    n_combinations = len(param_matrix)
    
    for i in prange(n_combinations):
        if state_matrix[i, 4] == 1.0:
            continue # Skip if hit Margin Call
            
        risk_pct = param_matrix[i, 0]
        min_range = param_matrix[i, 1]
        tf_mins = int(param_matrix[i, 2])
        hour = int(param_matrix[i, 3])
        minute = int(param_matrix[i, 4])
        
        equity = state_matrix[i, 0]
        max_drawdown_pct = state_matrix[i, 1]
        total_trades = state_matrix[i, 2]
        total_wins = state_matrix[i, 3]
        
        for d in range(days_in_chunk):
            day_start_ns = chunk_start_ts_ns + (d * 86400000000000)
            target_offset_ns = ((hour - 7) * 3600 + minute * 60) * 1000000000
            
            t_start = day_start_ns + target_offset_ns
            t_end_candle = t_start + (tf_mins * 60 * 1000000000)
            t_day_end = t_start + 86400000000000
            
            idx_start = np.searchsorted(timestamps, t_start)
            idx_candle_end = np.searchsorted(timestamps, t_end_candle)
            idx_day_end = np.searchsorted(timestamps, t_day_end)
            
            if idx_candle_end > idx_start:
                candle_prices = prices[idx_start:idx_candle_end]
                high = np.max(candle_prices)
                low = np.min(candle_prices)
                range_val = high - low
                
                if range_val >= min_range:
                    risk_amount = equity * risk_pct
                    lot_size = risk_amount / range_val
                    if lot_size < 0.01: lot_size = 0.01
                    
                    tick_prices = prices[idx_candle_end:idx_day_end]
                    
                    if len(tick_prices) > 0:
                        profit, n_trades, status, t1_mdd, t2_mdd = simulate_ticks(
                            tick_prices, high, low, lot_size, range_val, low, high, 
                            spread, equity, leverage
                        )
                        
                        if n_trades > 0:
                            total_trades += n_trades
                            if status == 1: total_wins += 1
                            
                            eq_t1 = max(equity, 1e-6)
                            max_drawdown_pct = max(max_drawdown_pct, (t1_mdd / eq_t1) * 100)
                            if n_trades == 2:
                                eq_t2 = max(equity + profit, 1e-6) 
                                max_drawdown_pct = max(max_drawdown_pct, (t2_mdd / eq_t2) * 100)
                        
                        equity += profit
                        
                        if status == -2 or equity <= 0:
                            state_matrix[i, 4] = 1.0 # Flag Margin Call
                            break
                            
        state_matrix[i, 0] = equity
        state_matrix[i, 1] = max_drawdown_pct
        state_matrix[i, 2] = total_trades
        state_matrix[i, 3] = total_wins

def load_data_chunk_to_ram(start_utc, end_utc):
    """Loads a block of CSV data into Fast Numpy arrays"""
    required_months = set()
    current = start_utc
    while current <= end_utc:
        required_months.add(current.strftime("%Y-%m"))
        if current.month == 12: current = datetime(current.year + 1, 1, 1)
        else: current = datetime(current.year, current.month + 1, 1)
            
    all_files = glob.glob(os.path.join(DATA_PATH, "ETHUSDT-aggTrades-*.csv"))
    files = [f for f in sorted(all_files) if any(ym in f for ym in required_months)]
    
    df_list = []
    start_us = int(start_utc.timestamp() * 1_000_000)
    end_us = int(end_utc.timestamp() * 1_000_000)
    
    for f in files:
        try:
            chunks = pd.read_csv(f, usecols=[1, 5], names=['price', 'timestamp'], header=None, chunksize=2000000)
            for chunk in chunks:
                if chunk['timestamp'].iloc[-1] < start_us: continue
                if chunk['timestamp'].iloc[0] > end_us: break
                mask = (chunk['timestamp'] >= start_us) & (chunk['timestamp'] <= end_us)
                if mask.any():
                    filtered = chunk[mask].copy()
                    filtered['timestamp'] = pd.to_datetime(filtered['timestamp'], unit='us')
                    df_list.append(filtered)
        except Exception:
            pass
            
    if df_list:
        master_df = pd.concat(df_list, ignore_index=True).sort_values('timestamp')
        prices_array = master_df['price'].values.astype(np.float32)
        timestamps_array = master_df['timestamp'].values.astype(np.int64) 
        return prices_array, timestamps_array
    return None, None

def load_top_20_parameters():
    """Reads optimization_results.csv and formats the top 20 for Numba execution"""
    if not os.path.exists(RESULTS_FILE):
        print(f"❌ Error: '{RESULTS_FILE}' not found. Run optimize.py first.")
        return None, None
        
    df_opt = pd.read_csv(RESULTS_FILE).head(20)
    
    # Load parameters straight from the CSV (no string splitting needed)
    risk_clean = df_opt['Risk'].astype(float)
    hour_clean = df_opt['Hour'].astype(int)
    min_clean = df_opt['Min'].astype(int)
    
    param_matrix = np.zeros((len(df_opt), 5), dtype=np.float64)
    param_matrix[:, 0] = risk_clean
    param_matrix[:, 1] = df_opt['Min Range']
    param_matrix[:, 2] = df_opt['TF (m)']
    param_matrix[:, 3] = hour_clean
    param_matrix[:, 4] = min_clean
    
    return param_matrix, df_opt

if __name__ == "__main__":
    print(f"📥 Loading Top 20 Parameters from {RESULTS_FILE}...")
    param_matrix, display_df = load_top_20_parameters()
    
    if param_matrix is None:
        exit()
        
    total_combinations = len(param_matrix)
    
    # State Matrix -> [equity, max_dd, total_trades, total_wins, mc_flag]
    state_matrix = np.zeros((total_combinations, 5), dtype=np.float64)
    state_matrix[:, 0] = INITIAL_BALANCE
    
    start_date_obj = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date_obj = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    print("======================================================")
    print(f"🚀 FORWARD TESTING TOP {total_combinations} STRATEGIES")
    print(f"📅 Testing Period: {START_DATE} to {END_DATE}")
    print("======================================================")
    
    start_time = time.time()
    
    # Process the timeline in 30-day blocks
    current_chunk_start = start_date_obj
    while current_chunk_start <= end_date_obj:
        current_chunk_end = min(current_chunk_start + timedelta(days=29), end_date_obj)
        days_in_chunk = (current_chunk_end - current_chunk_start).days + 1
        
        print(f"⏳ Processing Block: {current_chunk_start.date()} to {current_chunk_end.date()}...")
        
        chunk_start_utc = current_chunk_start - timedelta(hours=WIB_OFFSET) - timedelta(days=1)
        chunk_end_utc = current_chunk_end + timedelta(days=2)
        
        prices_array, timestamps_array = load_data_chunk_to_ram(chunk_start_utc, chunk_end_utc)
        
        if prices_array is not None:
            chunk_start_ts_ns = int(current_chunk_start.timestamp() * 1000000000)
            
            run_fast_grid_search_chunk(
                param_matrix, state_matrix, prices_array, timestamps_array, 
                chunk_start_ts_ns, days_in_chunk, LEVERAGE, SPREAD_PIPS
            )
            
            del prices_array, timestamps_array
            gc.collect() 
        else:
            print(f"   ⚠️ No data found for this block. Skipping.")
            
        current_chunk_start = current_chunk_end + timedelta(days=1)
        
    end_time = time.time()
    print(f"\n⚡ Testing Finished in {round((end_time - start_time), 2)} seconds!")
    
    # Attach calculated results back to the display DataFrame
    display_df['TEST Net Profit ($)'] = (state_matrix[:, 0] - INITIAL_BALANCE).round(2)
    
    trades_mask = state_matrix[:, 2] > 0
    win_rates = np.zeros(total_combinations)
    win_rates[trades_mask] = (state_matrix[trades_mask, 3] / state_matrix[trades_mask, 2]) * 100.0
    display_df['TEST Win Rate (%)'] = win_rates.round(2)
    
    display_df['TEST Max DD (%)'] = state_matrix[:, 1].round(2)
    display_df['TEST Total Trades'] = state_matrix[:, 2].astype(int)
    
    # Handle rows that hit Margin Call
    mc_mask = state_matrix[:, 4] == 1.0
    display_df.loc[mc_mask, 'TEST Net Profit ($)'] = "MARGIN CALL"
    
    # Create formatted strings for final console output
    display_df['Hour:Min'] = display_df.apply(lambda row: f"{int(row['Hour']):02d}:{int(row['Min']):02d}", axis=1)
    display_df['Risk'] = display_df['Risk'].apply(lambda x: f"{float(x)*100:.0f}%")
    
    # Keep only the identifying info and the newly tested results
    cols_to_show = ['Risk', 'Min Range', 'TF (m)', 'Hour:Min', 'TEST Total Trades', 'TEST Win Rate (%)', 'TEST Max DD (%)', 'TEST Net Profit ($)']
    final_report = display_df[cols_to_show]
    
    print("\n🏆 RESULTS FOR NEW DATE RANGE:")
    print("=========================================================================================")
    print(final_report.to_string(index=False))