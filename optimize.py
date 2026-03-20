import pandas as pd
import numpy as np
from numba import njit
import os
import glob
from datetime import datetime, timedelta
import multiprocessing as mp
import time
import sys

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

# Optimization Constraints
MAX_DD_THRESHOLD = 50.0  # Maximum acceptable Drawdown (%)
# ==========================================

# Global variables to hold ultra-fast Numpy arrays in Worker RAM
worker_prices = None
worker_timestamps = None

def init_worker(shared_prices, shared_timestamps):
    """Initializes each CPU worker with the raw Numpy arrays"""
    global worker_prices, worker_timestamps
    worker_prices = shared_prices
    worker_timestamps = shared_timestamps

@njit
def simulate_ticks(ticks, buy_stop, sell_stop, lot_size_in, tp_distance, sl_price_buy, sl_price_sell, spread, current_balance, leverage):
    """Numba optimized tick simulator (Silent)"""
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
                if t1_margin > active_balance: return 0.0, 1, -2, i, i, 0.0, 0.0, t1_margin, -1, -1, 0.0, 0.0, 0.0
                pos_type = 1
                tp_price = buy_stop + tp_distance 
                sl_price = sl_price_buy - s2
                t1_entry, trades_count = i, 1
                buy_pending = False
            elif sell_pending and p <= sell_stop:
                entry_price = sell_stop - s2
                t1_margin = (entry_price * lot_size) / leverage
                if t1_margin > active_balance: return 0.0, 1, -2, i, i, 0.0, 0.0, t1_margin, -1, -1, 0.0, 0.0, 0.0
                pos_type = 2
                tp_price = sell_stop - tp_distance
                sl_price = sl_price_sell + s2
                t1_entry, trades_count = i, 1
                sell_pending = False
                
        elif pos_type == 1: 
            margin = (entry_price * lot_size) / leverage
            unrealized_p = (p - entry_price) * lot_size
            floating_equity = active_balance - margin + unrealized_p
            drawdown_dollars = active_balance - floating_equity
            
            if drawdown_dollars > 0:
                if trades_count == 1: t1_mdd = max(t1_mdd, drawdown_dollars)
                else: t2_mdd = max(t2_mdd, drawdown_dollars)
                
            if floating_equity <= 0:
                if trades_count == 1: return unrealized_p, 1, -2, t1_entry, i, unrealized_p, t1_mdd, t1_margin, -1, -1, 0.0, 0.0, 0.0
                else: return t1_p + unrealized_p, 2, -2, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, i, unrealized_p, t2_mdd, t2_margin
                
            if p >= (tp_price + s2): 
                res_p = (tp_price - entry_price) * lot_size
                if trades_count == 1:
                    t1_p, t1_exit = res_p, i
                    return t1_p, 1, 1, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, -1, -1, 0.0, 0.0, 0.0
                else:
                    t2_p, t2_exit = res_p, i
                    return t1_p + t2_p, 2, 1, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, t2_exit, t2_p, t2_mdd, t2_margin
            elif p <= sl_price: 
                res_p = (sl_price - entry_price) * lot_size
                if trades_count == 1:
                    t1_p, t1_exit = res_p, i
                    active_balance += res_p 
                    new_entry_price = sl_price - s2
                    new_lot_size = lot_size * 2.0
                    t2_margin = (new_entry_price * new_lot_size) / leverage
                    if t2_margin > active_balance: return t1_p, 2, -2, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, i, i, 0.0, 0.0, t2_margin
                    pos_type, entry_price, lot_size = 2, new_entry_price, new_lot_size
                    tp_price, sl_price = sl_price - tp_distance, buy_stop + s2
                    t2_entry, trades_count = i, 2
                else:
                    t2_p, t2_exit = res_p, i
                    return t1_p + t2_p, 2, -1, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, t2_exit, t2_p, t2_mdd, t2_margin
                    
        elif pos_type == 2: 
            margin = (entry_price * lot_size) / leverage
            unrealized_p = (entry_price - p) * lot_size
            floating_equity = active_balance - margin + unrealized_p
            drawdown_dollars = active_balance - floating_equity
            
            if drawdown_dollars > 0:
                if trades_count == 1: t1_mdd = max(t1_mdd, drawdown_dollars)
                else: t2_mdd = max(t2_mdd, drawdown_dollars)

            if floating_equity <= 0:
                if trades_count == 1: return unrealized_p, 1, -2, t1_entry, i, unrealized_p, t1_mdd, t1_margin, -1, -1, 0.0, 0.0, 0.0
                else: return t1_p + unrealized_p, 2, -2, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, i, unrealized_p, t2_mdd, t2_margin

            if p <= (tp_price - s2): 
                res_p = (entry_price - tp_price) * lot_size
                if trades_count == 1:
                    t1_p, t1_exit = res_p, i
                    return t1_p, 1, 1, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, -1, -1, 0.0, 0.0, 0.0
                else:
                    t2_p, t2_exit = res_p, i
                    return t1_p + t2_p, 2, 1, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, t2_exit, t2_p, t2_mdd, t2_margin
            elif p >= sl_price: 
                res_p = (entry_price - sl_price) * lot_size
                if trades_count == 1:
                    t1_p, t1_exit = res_p, i
                    active_balance += res_p 
                    new_entry_price = sl_price + s2
                    new_lot_size = lot_size * 2.0
                    t2_margin = (new_entry_price * new_lot_size) / leverage
                    if t2_margin > active_balance: return t1_p, 2, -2, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, i, i, 0.0, 0.0, t2_margin
                    pos_type, entry_price, lot_size = 1, new_entry_price, new_lot_size
                    tp_price, sl_price = sl_price + tp_distance, sell_stop - s2
                    t2_entry, trades_count = i, 2
                else:
                    t2_p, t2_exit = res_p, i
                    return t1_p + t2_p, 2, -1, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, t2_exit, t2_p, t2_mdd, t2_margin
                    
    return total_profit, trades_count, 0, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, t2_exit, t2_p, t2_mdd, t2_margin


def evaluate_parameters(params):
    """Runs a complete backtest using lightning-fast Numpy Binary Search."""
    global worker_prices, worker_timestamps 
    
    risk_pct = params['risk']
    min_range = params['range']
    tf_mins = params['tf']
    hour = params['hour']
    minute = params['minute']
    
    start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    equity = INITIAL_BALANCE
    max_drawdown_pct = 0.0
    total_wins, total_losses, total_trades = 0, 0, 0
    is_margin_call = False
    
    current_day = start_date
    
    while current_day <= end_date:
        # 1. Define Target Times
        utc_time_target = current_day - timedelta(hours=WIB_OFFSET) + timedelta(hours=hour, minutes=minute)
        utc_time_end_candle = utc_time_target + timedelta(minutes=tf_mins)
        day_end = utc_time_target + timedelta(days=1)
        
        # 2. Convert to datetime64 for fast numpy search
        t_start = np.datetime64(utc_time_target)
        t_end_candle = np.datetime64(utc_time_end_candle)
        t_day_end = np.datetime64(day_end)
        
        # 3. BINARY SEARCH: Finds the exact row index in microseconds
        idx_start = np.searchsorted(worker_timestamps, t_start)
        idx_candle_end = np.searchsorted(worker_timestamps, t_end_candle)
        idx_day_end = np.searchsorted(worker_timestamps, t_day_end)
        
        # 4. Slice the arrays (Instantaneous)
        candle_prices = worker_prices[idx_start : idx_candle_end]
        
        if len(candle_prices) >= 2:
            high = np.max(candle_prices)
            low = np.min(candle_prices)
            range_val = high - low
            
            if range_val >= min_range:
                risk_amount = equity * risk_pct
                lot_size = max(0.01, risk_amount / range_val) 
                
                tick_prices = worker_prices[idx_candle_end : idx_day_end]
                
                if len(tick_prices) > 0:
                    res = simulate_ticks(
                        tick_prices, high, low, lot_size, range_val, low, high, 
                        SPREAD_PIPS, equity, LEVERAGE
                    )
                    profit, n_trades, status, t1_ent, t1_ex, t1_p, t1_mdd, t1_margin, t2_ent, t2_ex, t2_p, t2_mdd, t2_margin = res
                    
                    if n_trades > 0:
                        total_trades += n_trades
                        if t1_ent != -1:
                            if t1_p > 0 and status != -2: total_wins += 1
                            else: total_losses += 1
                            max_drawdown_pct = max(max_drawdown_pct, (t1_mdd / max(equity, 1e-6)) * 100)
                            
                        if t2_ent != -1:
                            if t2_p > 0 and status != -2: total_wins += 1
                            else: total_losses += 1
                            eq_t2 = max(equity + t1_p, 1e-6)
                            max_drawdown_pct = max(max_drawdown_pct, (t2_mdd / eq_t2) * 100)
                    
                    equity += profit
                    
                    if status == -2 or equity <= 0:
                        is_margin_call = True
                        break 
        
        current_day += timedelta(days=1)
        
    net_profit = equity - INITIAL_BALANCE
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    
    return {
        'Risk': risk_pct,
        'Min Range': min_range,
        'TF (m)': tf_mins,
        'Hour': hour,
        'Min': minute,
        'Net Profit ($)': round(net_profit, 2),
        'Win Rate (%)': round(win_rate, 2),
        'Max DD (%)': round(max_drawdown_pct, 2),
        'Total Trades': total_trades,
        'MC': is_margin_call
    }

def generate_parameter_grid():
    """Generates all valid combinations of parameters."""
    risks = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    ranges = list(range(5, 21)) # 5 to 20
    timeframes = [15, 30, 60]
    hours = list(range(24)) # 0 to 23
    
    combinations = []
    for rsk in risks:
        for rng in ranges:
            for tf in timeframes:
                for hr in hours:
                    # Generate minutes based on timeframe steps
                    minutes = list(range(0, 60, tf))
                    for mn in minutes:
                        combinations.append({
                            'risk': rsk,
                            'range': rng,
                            'tf': tf,
                            'hour': hr,
                            'minute': mn
                        })
    return combinations

def preload_data_to_ram(start_date_str, end_date_str):
    """Loads CSVs from disk ONCE, converts to pure Numpy arrays for speed"""
    print(f"💽 Pre-loading Data into RAM for {start_date_str} to {end_date_str}...")
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Preload slightly wider margin to cover all timezones/hours
    start_utc = start_date - timedelta(hours=WIB_OFFSET) - timedelta(days=1)
    end_utc = end_date + timedelta(days=2) 
    
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
        print(f"  -> Loading {os.path.basename(f)}...")
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
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if df_list:
        master_df = pd.concat(df_list)
        # Ensure strict chronological sorting for Binary Search to work
        master_df = master_df.sort_values('timestamp')
        
        # Convert to raw Numpy Arrays (The Secret to the Speed)
        prices_array = master_df['price'].values
        timestamps_array = master_df['timestamp'].values # Stored natively as datetime64[ns]
        
        print(f"✅ Pre-loaded {len(master_df):,} rows into RAM.\n")
        return prices_array, timestamps_array
    else:
        print("❌ Failed to load any data.")
        return None, None

if __name__ == "__main__":
    # 1. Load Data Once into Raw Arrays
    prices_array, timestamps_array = preload_data_to_ram(START_DATE, END_DATE)
    
    if prices_array is None:
        exit()

    # 2. Setup Grid
    grid = generate_parameter_grid()
    total_combinations = len(grid)
    
    print("======================================================")
    print(f"🚀 STARTING OPTIMIZATION OVER {total_combinations} COMBINATIONS")
    print(f"⚡ Running with Binary Search across {mp.cpu_count()} CPU cores...")
    print("======================================================")
    
    start_time = time.time()
    results = []
    
    # 3. Process via RAM-initialized Multiprocessing WITH LIVE PROGRESS BAR
    with mp.Pool(processes=mp.cpu_count(), initializer=init_worker, initargs=(prices_array, timestamps_array)) as pool:
        # imap_unordered allows us to process results the moment they finish instead of waiting for all of them
        for i, res in enumerate(pool.imap_unordered(evaluate_parameters, grid), 1):
            results.append(res)
            
            # Calculate live elapsed time
            elapsed = time.time() - start_time
            
            # '\r' pushes the cursor back to the start of the line so it updates in place!
            print(f"\r⏳ Running... [{i}/{total_combinations}] | Elapsed Time: {elapsed:.2f} seconds", end="", flush=True)
            
    end_time = time.time()
    print(f"\n\n⚡ Optimization Completely Finished in {round((end_time - start_time), 2)} seconds!")
    
    # 4. Process Results
    df_results = pd.DataFrame(results)
    
    filtered_df = df_results[
        (df_results['MC'] == False) & 
        (df_results['Max DD (%)'] <= MAX_DD_THRESHOLD) &
        (df_results['Total Trades'] > 0)
    ].copy()
    
    if filtered_df.empty:
        print("\n❌ No parameter combinations survived the filters (All hit Margin Call, exceeded Max Drawdown, or took 0 trades).")
    else:
        filtered_df = filtered_df.sort_values(by='Net Profit ($)', ascending=False).reset_index(drop=True)
        filtered_df = filtered_df.drop(columns=['MC'])
        
        print(f"\n✅ Found {len(filtered_df)} valid combinations.")
        print(f"🏆 TOP 20 BEST PARAMETER COMBINATIONS (Max DD <= {MAX_DD_THRESHOLD}%):")
        print("=========================================================================================")
        
        display_df = filtered_df.head(20).copy()
        display_df['Risk'] = display_df['Risk'].apply(lambda x: f"{x*100:.0f}%")
        display_df['Hour:Min'] = display_df.apply(lambda row: f"{int(row['Hour']):02d}:{int(row['Min']):02d}", axis=1)
        
        cols = ['Risk', 'Min Range', 'TF (m)', 'Hour:Min', 'Total Trades', 'Win Rate (%)', 'Max DD (%)', 'Net Profit ($)']
        display_df = display_df[cols]
        
        print(display_df.to_string(index=False))
        
        filtered_df.to_csv("optimization_results.csv", index=False)
        print("\n💾 Full list of valid parameters saved to 'optimization_results.csv'")