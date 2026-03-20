import pandas as pd
import numpy as np
from numba import njit, prange
import os
import glob
from datetime import datetime, timedelta, timezone
import time
import gc

# ==========================================
# WALK-FORWARD ANALYSIS CONFIGURATION
# ==========================================
DATA_PATH = r"d:\repos\ethusd\data\raw"

TARGET_DATE = "2026-01-10"   # The ONE future day we are trying to predict
MAX_DAYS_BACK = 30           # Max historical lookback (1 to 9 days back)
TOP_N_PARAMS = 5             # Take Top 5 parameters from each lookback period

INITIAL_BALANCE = 100.0
LEVERAGE = 400
SPREAD_PIPS = 0.5
WIB_OFFSET = 7

MAX_DD_THRESHOLD = 50.0  
# ==========================================

@njit
def simulate_ticks(ticks, buy_stop, sell_stop, lot_size_in, tp_distance, sl_price_buy, sl_price_sell, spread, current_balance, leverage):
    """Core tick simulator with End of Day Force Close (Float64 Accurate)"""
    pos_type = 0 
    entry_price, tp_price, sl_price = 0.0, 0.0, 0.0
    lot_size = lot_size_in
    trades_count = 0
    t1_p, t1_mdd, t1_margin = 0.0, 0.0, 0.0
    t2_p, t2_mdd, t2_margin = 0.0, 0.0, 0.0
    buy_pending, sell_pending = True, True
    active_balance = current_balance
    s2 = spread / 2.0
    
    for i in range(len(ticks)):
        p = ticks[i]
        if pos_type == 0:
            if buy_pending and p >= buy_stop:
                entry_price = buy_stop + s2
                t1_margin = (entry_price * lot_size) / leverage
                if t1_margin > active_balance: return 0.0, 1, -2, 0.0, 0.0, 0.0, 0.0
                pos_type = 1
                tp_price = buy_stop + tp_distance 
                sl_price = sl_price_buy - s2
                trades_count = 1
                buy_pending = False
            elif sell_pending and p <= sell_stop:
                entry_price = sell_stop - s2
                t1_margin = (entry_price * lot_size) / leverage
                if t1_margin > active_balance: return 0.0, 1, -2, 0.0, 0.0, 0.0, 0.0
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
                if trades_count == 1: return unrealized_p, 1, -2, unrealized_p, 0.0, t1_mdd, t2_mdd
                else: return t1_p + unrealized_p, 2, -2, t1_p, unrealized_p, t1_mdd, t2_mdd
                
            if p >= (tp_price + s2): 
                res_p = (tp_price - entry_price) * lot_size
                if trades_count == 1: return res_p, 1, 1, res_p, 0.0, t1_mdd, t2_mdd
                else: return t1_p + res_p, 2, 1, t1_p, res_p, t1_mdd, t2_mdd
            elif p <= sl_price: 
                res_p = (sl_price - entry_price) * lot_size
                if trades_count == 1:
                    t1_p = res_p
                    active_balance += res_p 
                    new_entry_price = sl_price - s2
                    new_lot_size = lot_size * 2.0
                    t2_margin = (new_entry_price * new_lot_size) / leverage
                    if t2_margin > active_balance: return t1_p, 2, -2, t1_p, 0.0, t1_mdd, t2_mdd
                    pos_type, entry_price, lot_size = 2, new_entry_price, new_lot_size
                    tp_price, sl_price = sl_price - tp_distance, buy_stop + s2
                    trades_count = 2
                else:
                    return t1_p + res_p, 2, -1, t1_p, res_p, t1_mdd, t2_mdd
                    
        elif pos_type == 2: 
            unrealized_p = (entry_price - p) * lot_size
            floating_equity = active_balance - ((entry_price * lot_size) / leverage) + unrealized_p
            drawdown_dollars = active_balance - floating_equity
            
            if drawdown_dollars > 0:
                if trades_count == 1: t1_mdd = max(t1_mdd, drawdown_dollars)
                else: t2_mdd = max(t2_mdd, drawdown_dollars)

            if floating_equity <= 0:
                if trades_count == 1: return unrealized_p, 1, -2, unrealized_p, 0.0, t1_mdd, t2_mdd
                else: return t1_p + unrealized_p, 2, -2, t1_p, unrealized_p, t1_mdd, t2_mdd

            if p <= (tp_price - s2): 
                res_p = (entry_price - tp_price) * lot_size
                if trades_count == 1: return res_p, 1, 1, res_p, 0.0, t1_mdd, t2_mdd
                else: return t1_p + res_p, 2, 1, t1_p, res_p, t1_mdd, t2_mdd
            elif p >= sl_price: 
                res_p = (entry_price - sl_price) * lot_size
                if trades_count == 1:
                    t1_p = res_p
                    active_balance += res_p 
                    new_entry_price = sl_price + s2
                    new_lot_size = lot_size * 2.0
                    t2_margin = (new_entry_price * new_lot_size) / leverage
                    if t2_margin > active_balance: return t1_p, 2, -2, t1_p, 0.0, t1_mdd, t2_mdd
                    pos_type, entry_price, lot_size = 1, new_entry_price, new_lot_size
                    tp_price, sl_price = sl_price + tp_distance, sell_stop - s2
                    trades_count = 2
                else:
                    return t1_p + res_p, 2, -1, t1_p, res_p, t1_mdd, t2_mdd

    if pos_type == 1:
        res_p = (p - entry_price) * lot_size
        if trades_count == 1: return res_p, 1, 0, res_p, 0.0, t1_mdd, t2_mdd
        else: return t1_p + res_p, 2, 0, t1_p, res_p, t1_mdd, t2_mdd
    elif pos_type == 2:
        res_p = (entry_price - p) * lot_size
        if trades_count == 1: return res_p, 1, 0, res_p, 0.0, t1_mdd, t2_mdd
        else: return t1_p + res_p, 2, 0, t1_p, res_p, t1_mdd, t2_mdd
                    
    return 0.0, trades_count, 0, 0.0, 0.0, 0.0, 0.0

@njit(parallel=True)
def run_fast_grid_search(param_matrix, prices, timestamps, start_ts_ns, days_in_chunk, leverage, spread, wib_offset):
    n_combinations = len(param_matrix)
    state_matrix = np.zeros((n_combinations, 5), dtype=np.float64)
    state_matrix[:, 0] = INITIAL_BALANCE # Set initial equity
    
    for i in prange(n_combinations):
        risk_pct = param_matrix[i, 0]
        min_range = param_matrix[i, 1]
        tf_mins = int(param_matrix[i, 2])
        hour = int(param_matrix[i, 3])
        minute = int(param_matrix[i, 4])
        
        equity = state_matrix[i, 0]
        max_drawdown_pct = 0.0
        total_trades = 0
        total_wins = 0
        
        for d in range(days_in_chunk):
            day_start_ns = start_ts_ns + (d * 86400000000000)
            target_offset_ns = ((hour - wib_offset) * 3600 + minute * 60) * 1000000000
            
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
                        profit, n_trades, status, t1_p, t2_p, t1_mdd, t2_mdd = simulate_ticks(
                            tick_prices, high, low, lot_size, range_val, low, high, 
                            spread, equity, leverage
                        )
                        
                        if n_trades > 0:
                            total_trades += n_trades
                            if t1_p > 0: total_wins += 1
                            if n_trades == 2 and t2_p > 0: total_wins += 1
                            
                            eq_t1 = max(equity, 1e-6)
                            max_drawdown_pct = max(max_drawdown_pct, (t1_mdd / eq_t1) * 100)
                            if n_trades == 2:
                                eq_t2 = max(equity + t1_p, 1e-6) 
                                max_drawdown_pct = max(max_drawdown_pct, (t2_mdd / eq_t2) * 100)
                        
                        equity += profit
                        
                        if status == -2 or equity <= 0:
                            state_matrix[i, 4] = 1.0 # Margin Call
                            break
                            
        state_matrix[i, 0] = equity
        state_matrix[i, 1] = max_drawdown_pct
        state_matrix[i, 2] = total_trades
        state_matrix[i, 3] = total_wins
        
    return state_matrix

def load_data_chunk_to_ram(start_utc, end_utc):
    required_months = set()
    current = start_utc
    while current <= end_utc:
        required_months.add(current.strftime("%Y-%m"))
        if current.month == 12: current = datetime(current.year + 1, 1, 1)
        else: current = datetime(current.year, current.month + 1, 1)
            
    all_files = glob.glob(os.path.join(DATA_PATH, "ETHUSDT-aggTrades-*.csv"))
    files = [f for f in sorted(all_files) if any(ym in f for ym in required_months)]
    
    df_list = []
    start_us = int(start_utc.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
    end_us = int(end_utc.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
    
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
        prices_array = master_df['price'].values.astype(np.float64) 
        timestamps_array = master_df['timestamp'].values.astype(np.int64) 
        return prices_array, timestamps_array
    return None, None

def generate_parameter_grid():
    risks = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    ranges = list(range(5, 21)) 
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
    target_dt = datetime.strptime(TARGET_DATE, "%Y-%m-%d")
    
    print("======================================================")
    print(f"🚀 WALK-FORWARD OPTIMIZATION & TESTING")
    print(f"🎯 Target Prediction Date: {TARGET_DATE}")
    print(f"🕒 Historical Lookback  : up to {MAX_DAYS_BACK - 1} Days")
    print("======================================================")
    
    # 1. LOAD ALL NECESSARY DATA ONCE
    data_start_utc = target_dt - timedelta(days=MAX_DAYS_BACK + 2)
    data_end_utc = target_dt + timedelta(days=2)
    
    print(f"💽 Loading continuous RAM window from {data_start_utc.date()} to {data_end_utc.date()}...")
    prices_array, timestamps_array = load_data_chunk_to_ram(data_start_utc, data_end_utc)
    
    if prices_array is None:
        print("❌ No data available. Exiting.")
        exit()
        
    param_matrix = generate_parameter_grid()
    total_combinations = len(param_matrix)
    
    overall_start_time = time.time()
    
    # Dictionary to keep track of Best Params and what Lookback Window found them
    # Key: (risk, range, tf, hr, min) -> Value: list of lookbacks e.g., [1, 4, 8]
    best_params_tracker = {}
    
    # 2. RUN HISTORICAL OPTIMIZATIONS
    print(f"\n🔍 STEP 1: FINDING BEST PARAMETERS IN {MAX_DAYS_BACK - 1} HISTORICAL WINDOWS")
    for lookback in range(1, MAX_DAYS_BACK):
        step_start_time = time.time()
        
        # Calculate Start and End for this specific historical lookback
        opt_start_dt = target_dt - timedelta(days=lookback)
        opt_start_ts_ns = int(opt_start_dt.replace(tzinfo=timezone.utc).timestamp() * 1_000_000_000)
        
        # Run Grid Search
        state_matrix = run_fast_grid_search(
            param_matrix, prices_array, timestamps_array, 
            opt_start_ts_ns, lookback, LEVERAGE, SPREAD_PIPS, WIB_OFFSET
        )
        
        # Compile Results
        results_df = pd.DataFrame(param_matrix, columns=['Risk', 'Min Range', 'TF (m)', 'Hour', 'Min'])
        results_df['Net Profit ($)'] = state_matrix[:, 0] - INITIAL_BALANCE
        results_df['Max DD (%)'] = state_matrix[:, 1]
        results_df['Total Trades'] = state_matrix[:, 2]
        results_df['MC'] = state_matrix[:, 4]
        
        # Filter & Sort
        filtered_df = results_df[
            (results_df['MC'] == 0.0) & 
            (results_df['Max DD (%)'] <= MAX_DD_THRESHOLD) &
            (results_df['Total Trades'] > 0)
        ].sort_values(by='Net Profit ($)', ascending=False)
        
        # Extract Top N
        top_n_df = filtered_df.head(TOP_N_PARAMS)
        
        # Save to our deduplication tracker
        for _, row in top_n_df.iterrows():
            p_tuple = (row['Risk'], row['Min Range'], row['TF (m)'], row['Hour'], row['Min'])
            if p_tuple not in best_params_tracker:
                best_params_tracker[p_tuple] = []
            best_params_tracker[p_tuple].append(lookback)
            
        step_time = time.time() - step_start_time
        print(f"   ✅ [Lookback {lookback} Days] Found top {len(top_n_df)} params in {step_time:.2f}s")
        
    unique_fwd_params = list(best_params_tracker.keys())
    num_unique = len(unique_fwd_params)
    print(f"\n📊 Extracted {num_unique} unique high-performing parameter sets across all historical windows.")
    
    # 3. RUN FORWARD TEST ON THE TARGET DATE
    print(f"\n🎯 STEP 2: FORWARD TESTING UNIQUE PARAMETERS ON TARGET DATE ({TARGET_DATE})")
    
    fwd_param_matrix = np.array(unique_fwd_params, dtype=np.float64)
    target_ts_ns = int(target_dt.replace(tzinfo=timezone.utc).timestamp() * 1_000_000_000)
    
    # We forward test exactly 1 Day (The target date)
    fwd_state_matrix = run_fast_grid_search(
        fwd_param_matrix, prices_array, timestamps_array, 
        target_ts_ns, 1, LEVERAGE, SPREAD_PIPS, WIB_OFFSET
    )
    
    # 4. PROCESS FINAL RANKING
    final_results = []
    for i in range(num_unique):
        p_tuple = unique_fwd_params[i]
        
        # Identify which lookback periods found this parameter
        lookbacks = sorted(best_params_tracker[p_tuple])
        lookback_str = ", ".join(map(str, lookbacks))
        
        profit = fwd_state_matrix[i, 0] - INITIAL_BALANCE
        max_dd = fwd_state_matrix[i, 1]
        total_trades = fwd_state_matrix[i, 2]
        total_wins = fwd_state_matrix[i, 3]
        mc_flag = fwd_state_matrix[i, 4]
        
        win_rate = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
        
        final_results.append({
            'Risk': f"{p_tuple[0]*100:.0f}%",
            'Min Range': p_tuple[1],
            'TF (m)': int(p_tuple[2]),
            'Hour:Min': f"{int(p_tuple[3]):02d}:{int(p_tuple[4]):02d}",
            'Lookbacks Used': lookback_str,
            'FWD Trades': int(total_trades),
            'FWD Win %': round(win_rate, 2),
            'FWD Max DD (%)': round(max_dd, 2),
            'FWD Profit ($)': round(profit, 2),
            'MC': mc_flag
        })
        
    df_final = pd.DataFrame(final_results)
    
    # Filter Final Forward Results (No Margin Calls, Strict DD)
    df_final_filtered = df_final[
        (df_final['MC'] == 0.0) & 
        (df_final['FWD Max DD (%)'] <= MAX_DD_THRESHOLD) &
        (df_final['FWD Trades'] > 0)
    ].copy()
    
    df_final_filtered = df_final_filtered.sort_values(by='FWD Profit ($)', ascending=False).reset_index(drop=True)
    df_final_filtered = df_final_filtered.drop(columns=['MC'])
    
    overall_time = time.time() - overall_start_time
    
    print("\n🏆 FINAL WALK-FORWARD RESULTS RANKING")
    print("==========================================================================================================")
    if df_final_filtered.empty:
        print("❌ No parameters survived the Forward Test on the Target Date without Margin Call or extreme Drawdown.")
    else:
        print(df_final_filtered.to_string(index=False))
        
        # Export to CSV
        output_file = f"walk_forward_target_{TARGET_DATE}.csv"
        df_final_filtered.to_csv(output_file, index=False)
        print(f"\n💾 Saved detailed ranking to '{output_file}'")
        
    print(f"\n⚡ Total WFA execution time: {overall_time:.2f} seconds")