import pandas as pd
import numpy as np
from numba import njit, prange
import os
import glob
from datetime import datetime, timedelta
import time
import gc

# ==========================================
# OPTIMIZER CONFIGURATION
# ==========================================
DATA_PATH = r"d:\repos\ethusd\data\raw"
START_DATE = "2025-01-02"
END_DATE = "2026-02-01"
INITIAL_BALANCE = 100.0
LEVERAGE = 400
SPREAD_PIPS = 0.5
WIB_OFFSET = 7

MAX_DD_THRESHOLD = 50.0  
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
    """
    Executes grid search iteratively. 
    Maintains memory (Equity, Drawdown) in state_matrix between chunk loads.
    """
    n_combinations = len(param_matrix)
    
    # prange instructs Numba to divide this loop across all CPU cores instantly
    for i in prange(n_combinations):
        if state_matrix[i, 4] == 1.0:
            continue # Skip if this param combination previously hit a Margin Call
            
        risk_pct = param_matrix[i, 0]
        min_range = param_matrix[i, 1]
        tf_mins = int(param_matrix[i, 2])
        hour = int(param_matrix[i, 3])
        minute = int(param_matrix[i, 4])
        
        # Load State from previous months
        equity = state_matrix[i, 0]
        max_drawdown_pct = state_matrix[i, 1]
        total_trades = state_matrix[i, 2]
        total_wins = state_matrix[i, 3]
        
        for d in range(days_in_chunk):
            # Calculate target timestamps in nanoseconds (pandas native format)
            day_start_ns = chunk_start_ts_ns + (d * 86400000000000) # 86.4 Trillion nanos in a day
            target_offset_ns = ((hour - 7) * 3600 + minute * 60) * 1000000000
            
            t_start = day_start_ns + target_offset_ns
            t_end_candle = t_start + (tf_mins * 60 * 1000000000)
            t_day_end = t_start + 86400000000000
            
            # Fast Binary Search
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
                            
                            # DD calcs
                            eq_t1 = max(equity, 1e-6)
                            max_drawdown_pct = max(max_drawdown_pct, (t1_mdd / eq_t1) * 100)
                            if n_trades == 2:
                                eq_t2 = max(equity + profit, 1e-6) 
                                max_drawdown_pct = max(max_drawdown_pct, (t2_mdd / eq_t2) * 100)
                        
                        equity += profit
                        
                        if status == -2 or equity <= 0:
                            state_matrix[i, 4] = 1.0 # Flag Margin Call
                            break
                            
        # Save state to carry over to the next month
        state_matrix[i, 0] = equity
        state_matrix[i, 1] = max_drawdown_pct
        state_matrix[i, 2] = total_trades
        state_matrix[i, 3] = total_wins

def load_data_chunk_to_ram(start_utc, end_utc):
    """Loads a small block of CSV data to avoid MemoryErrors"""
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
        master_df = pd.concat(df_list, ignore_index=True)
        # Sort is safe here because we only load 30 days maximum
        master_df = master_df.sort_values('timestamp')
        
        # Float32 uses 50% less RAM
        prices_array = master_df['price'].values.astype(np.float32)
        # Natively cast to Int64 nanoseconds for Numba Binary Search
        timestamps_array = master_df['timestamp'].values.astype(np.int64) 
        
        return prices_array, timestamps_array
    return None, None

def generate_parameter_grid():
    risks = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    ranges = list(range(5, 21)) # 16 items
    timeframes = [15, 30, 60]
    hours = list(range(24))
    
    param_list = []
    for rsk in risks:
        for rng in ranges:
            for tf in timeframes:
                for hr in hours:
                    for mn in range(0, 60, tf):
                        param_list.append([rsk, rng, tf, hr, mn])
                        
    return np.array(param_list, dtype=np.float64)

if __name__ == "__main__":
    param_matrix = generate_parameter_grid()
    total_combinations = len(param_matrix)
    
    # State Matrix to keep memory across chunks -> [equity, max_dd, total_trades, total_wins, mc_flag]
    state_matrix = np.zeros((total_combinations, 5), dtype=np.float64)
    state_matrix[:, 0] = INITIAL_BALANCE
    
    start_date_obj = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date_obj = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    print("======================================================")
    print(f"🚀 OPTIMIZING {total_combinations} COMBINATIONS (MEMORY SAFE MODE)")
    print("======================================================")
    
    start_time = time.time()
    
    # Block Loop - Process the timeline in 30-day blocks
    current_chunk_start = start_date_obj
    while current_chunk_start <= end_date_obj:
        current_chunk_end = min(current_chunk_start + timedelta(days=29), end_date_obj)
        days_in_chunk = (current_chunk_end - current_chunk_start).days + 1
        
        print(f"⏳ Processing Block: {current_chunk_start.date()} to {current_chunk_end.date()}...")
        
        # Padding bounds
        chunk_start_utc = current_chunk_start - timedelta(hours=WIB_OFFSET) - timedelta(days=1)
        chunk_end_utc = current_chunk_end + timedelta(days=2)
        
        # 1. Load RAM chunk safely
        prices_array, timestamps_array = load_data_chunk_to_ram(chunk_start_utc, chunk_end_utc)
        
        if prices_array is not None:
            chunk_start_ts_ns = int(current_chunk_start.timestamp() * 1000000000)
            
            # 2. Run Numba Evaluation
            run_fast_grid_search_chunk(
                param_matrix, state_matrix, prices_array, timestamps_array, 
                chunk_start_ts_ns, days_in_chunk, LEVERAGE, SPREAD_PIPS
            )
            
            # 3. Aggressive RAM Cleanup to prevent crash before next loop
            del prices_array, timestamps_array
            gc.collect() 
        else:
            print(f"   ⚠️ No data found for this block. Skipping.")
            
        current_chunk_start = current_chunk_end + timedelta(days=1)
        
    end_time = time.time()
    print(f"\n⚡ Optimization Finished in {round((end_time - start_time) / 60, 2)} minutes!")
    
    # Process Output
    results_df = pd.DataFrame(param_matrix, columns=['Risk', 'Min Range', 'TF (m)', 'Hour', 'Min'])
    results_df['Net Profit ($)'] = state_matrix[:, 0] - INITIAL_BALANCE
    
    # Handle Div-by-Zero warning mathematically
    trades_mask = state_matrix[:, 2] > 0
    win_rates = np.zeros(total_combinations)
    win_rates[trades_mask] = (state_matrix[trades_mask, 3] / state_matrix[trades_mask, 2]) * 100.0
    
    results_df['Win Rate (%)'] = win_rates
    results_df['Max DD (%)'] = state_matrix[:, 1]
    results_df['Total Trades'] = state_matrix[:, 2]
    results_df['MC'] = state_matrix[:, 4]
    
    # Filter constraints
    filtered_df = results_df[
        (results_df['MC'] == 0.0) & 
        (results_df['Max DD (%)'] <= MAX_DD_THRESHOLD) &
        (results_df['Total Trades'] > 0)
    ].copy()
    
    if filtered_df.empty:
        print("\n❌ No combinations survived the filters.")
    else:
        filtered_df = filtered_df.sort_values(by='Net Profit ($)', ascending=False).reset_index(drop=True)
        filtered_df = filtered_df.drop(columns=['MC'])
        
        print(f"\n✅ Found {len(filtered_df)} valid combinations.")
        print(f"🏆 TOP 20 BEST PARAMETERS (Max DD <= {MAX_DD_THRESHOLD}%):")
        
        display_df = filtered_df.head(20).copy()
        display_df['Risk'] = display_df['Risk'].apply(lambda x: f"{x*100:.0f}%")
        display_df['Hour:Min'] = display_df.apply(lambda row: f"{int(row['Hour']):02d}:{int(row['Min']):02d}", axis=1)
        display_df['Net Profit ($)'] = display_df['Net Profit ($)'].round(2)
        display_df['Win Rate (%)'] = display_df['Win Rate (%)'].round(2)
        display_df['Max DD (%)'] = display_df['Max DD (%)'].round(2)
        display_df['Total Trades'] = display_df['Total Trades'].astype(int)
        
        cols = ['Risk', 'Min Range', 'TF (m)', 'Hour:Min', 'Total Trades', 'Win Rate (%)', 'Max DD (%)', 'Net Profit ($)']
        display_df = display_df[cols]
        print(display_df.to_string(index=False))
        
        filtered_df.to_csv("optimization_results.csv", index=False)