import pandas as pd
import numpy as np
from numba import njit
import os
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ==========================================
# STRATEGY CONFIGURATION & PARAMETERS
# ==========================================
CONFIG = {
    # Trading Parameters
    "initial_balance": 100.0,
    "leverage": 400,
    "risk_percent": 0.10,       # 10% risk per trade
    "min_range_pips": 5.0,      # Minimum candle range to trade
    "spread_pips": 0.5,         # Broker spread
    
    # Time & Setup Parameters
    "setup_hour": 21,            # Local hour of the setup candle (e.g., 5 for 05:xx)
    "setup_minute": 30,         # Local minute of the setup candle (e.g., 15 for xx:15)
    "timeframe_minutes": 15,    # Duration of the setup candle in minutes
    "wib_offset": 7,            # Local Timezone Offset (WIB is UTC+7)
    
    # Backtest Environment
    "data_path": r"d:\repos\ethusd\data\raw",
    "start_date": "2026-01-01", # Format: YYYY-MM-DD
    "end_date": "2026-02-01"    # Format: YYYY-MM-DD
}
# ==========================================

@njit
def simulate_ticks(ticks, buy_stop, sell_stop, lot_size_in, tp_distance, sl_price_buy, sl_price_sell, spread, current_balance, leverage):
    """
    Simultaneously watches 2 stop orders.
    Returns: (total_profit, num_trades, status, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, t2_exit, t2_p, t2_mdd, t2_margin)
    status: 1 (Win), -1 (Loss), 0 (No trade), -2 (Margin Call)
    """
    pos_type = 0 # 0: none, 1: long, 2: short
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    lot_size = lot_size_in
    
    trades_count = 0
    total_profit = 0.0
    
    t1_entry, t1_exit, t1_p, t1_mdd, t1_margin = -1, -1, 0.0, 0.0, 0.0
    t2_entry, t2_exit, t2_p, t2_mdd, t2_margin = -1, -1, 0.0, 0.0, 0.0
    
    buy_pending = True
    sell_pending = True
    
    active_balance = current_balance
    
    # Spread adjustments
    s2 = spread / 2.0
    
    for i in range(len(ticks)):
        p = ticks[i]
        
        if pos_type == 0:
            if buy_pending and p >= buy_stop:
                entry_price = buy_stop + s2
                t1_margin = (entry_price * lot_size) / leverage
                
                # MARGIN CALL CHECK 1: Insufficient balance to open Trade 1
                if t1_margin > active_balance:
                    return 0.0, 1, -2, i, i, 0.0, 0.0, t1_margin, -1, -1, 0.0, 0.0, 0.0
                
                pos_type = 1
                tp_price = buy_stop + tp_distance # Target is relative to original stop
                sl_price = sl_price_buy - s2
                t1_entry, trades_count = i, 1
                buy_pending = False
                
            elif sell_pending and p <= sell_stop:
                entry_price = sell_stop - s2
                t1_margin = (entry_price * lot_size) / leverage
                
                # MARGIN CALL CHECK 1: Insufficient balance to open Trade 1
                if t1_margin > active_balance:
                    return 0.0, 1, -2, i, i, 0.0, 0.0, t1_margin, -1, -1, 0.0, 0.0, 0.0
                
                pos_type = 2
                tp_price = sell_stop - tp_distance
                sl_price = sl_price_sell + s2
                t1_entry, trades_count = i, 1
                sell_pending = False
                
        elif pos_type == 1: # Long
            # Calculate floating equity
            margin = (entry_price * lot_size) / leverage
            unrealized_p = (p - entry_price) * lot_size
            floating_equity = active_balance - margin + unrealized_p
            
            # Track unrealized drawdown
            drawdown_dollars = active_balance - floating_equity
            if drawdown_dollars > 0:
                if trades_count == 1: t1_mdd = max(t1_mdd, drawdown_dollars)
                else: t2_mdd = max(t2_mdd, drawdown_dollars)
                
            # MARGIN CALL CHECK 2: Floating equity drops to 0 or below
            if floating_equity <= 0:
                if trades_count == 1:
                    return unrealized_p, 1, -2, t1_entry, i, unrealized_p, t1_mdd, t1_margin, -1, -1, 0.0, 0.0, 0.0
                else:
                    return t1_p + unrealized_p, 2, -2, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, i, unrealized_p, t2_mdd, t2_margin
                
            if p >= (tp_price + s2): # Exit TP
                res_p = (tp_price - entry_price) * lot_size
                if trades_count == 1:
                    t1_p, t1_exit = res_p, i
                    return t1_p, 1, 1, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, -1, -1, 0.0, 0.0, 0.0
                else:
                    t2_p, t2_exit = res_p, i
                    return t1_p + t2_p, 2, 1, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, t2_exit, t2_p, t2_mdd, t2_margin
            
            elif p <= sl_price: # Exit SL
                res_p = (sl_price - entry_price) * lot_size
                if trades_count == 1:
                    t1_p, t1_exit = res_p, i
                    active_balance += res_p # Update balance for reversal trade
                    
                    new_entry_price = sl_price - s2
                    new_lot_size = lot_size * 2.0
                    t2_margin = (new_entry_price * new_lot_size) / leverage
                    
                    # MARGIN CALL CHECK 1: Insufficient balance to open Trade 2 (Reversal)
                    if t2_margin > active_balance:
                        return t1_p, 2, -2, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, i, i, 0.0, 0.0, t2_margin
                    
                    # Reverse to Short
                    pos_type, entry_price = 2, new_entry_price
                    lot_size = new_lot_size
                    tp_price, sl_price = sl_price - tp_distance, buy_stop + s2
                    t2_entry, trades_count = i, 2
                else:
                    t2_p, t2_exit = res_p, i
                    return t1_p + t2_p, 2, -1, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, t2_exit, t2_p, t2_mdd, t2_margin
                    
        elif pos_type == 2: # Short
            # Calculate floating equity
            margin = (entry_price * lot_size) / leverage
            unrealized_p = (entry_price - p) * lot_size
            floating_equity = active_balance - margin + unrealized_p
            
            # Track unrealized drawdown
            drawdown_dollars = active_balance - floating_equity
            if drawdown_dollars > 0:
                if trades_count == 1: t1_mdd = max(t1_mdd, drawdown_dollars)
                else: t2_mdd = max(t2_mdd, drawdown_dollars)

            # MARGIN CALL CHECK 2: Floating equity drops to 0 or below
            if floating_equity <= 0:
                if trades_count == 1:
                    return unrealized_p, 1, -2, t1_entry, i, unrealized_p, t1_mdd, t1_margin, -1, -1, 0.0, 0.0, 0.0
                else:
                    return t1_p + unrealized_p, 2, -2, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, i, unrealized_p, t2_mdd, t2_margin

            if p <= (tp_price - s2): # Exit TP
                res_p = (entry_price - tp_price) * lot_size
                if trades_count == 1:
                    t1_p, t1_exit = res_p, i
                    return t1_p, 1, 1, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, -1, -1, 0.0, 0.0, 0.0
                else:
                    t2_p, t2_exit = res_p, i
                    return t1_p + t2_p, 2, 1, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, t2_exit, t2_p, t2_mdd, t2_margin
            
            elif p >= sl_price: # Exit SL
                res_p = (entry_price - sl_price) * lot_size
                if trades_count == 1:
                    t1_p, t1_exit = res_p, i
                    active_balance += res_p # Update balance for reversal trade
                    
                    new_entry_price = sl_price + s2
                    new_lot_size = lot_size * 2.0
                    t2_margin = (new_entry_price * new_lot_size) / leverage
                    
                    # MARGIN CALL CHECK 1: Insufficient balance to open Trade 2 (Reversal)
                    if t2_margin > active_balance:
                        return t1_p, 2, -2, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, i, i, 0.0, 0.0, t2_margin
                    
                    # Reverse to Long
                    pos_type, entry_price = 1, new_entry_price
                    lot_size = new_lot_size
                    tp_price, sl_price = sl_price + tp_distance, sell_stop - s2
                    t2_entry, trades_count = i, 2
                else:
                    t2_p, t2_exit = res_p, i
                    return t1_p + t2_p, 2, -1, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, t2_exit, t2_p, t2_mdd, t2_margin
                    
    return total_profit, trades_count, 0, t1_entry, t1_exit, t1_p, t1_mdd, t1_margin, t2_entry, t2_exit, t2_p, t2_mdd, t2_margin

def load_file_data(f, start_utc, end_utc):
    """Sequential loader to save memory."""
    df_list = []
    
    # UTC target range in microseconds
    start_us = int(start_utc.timestamp() * 1_000_000)
    end_us = int(end_utc.timestamp() * 1_000_000)
    
    print(f"Opening {os.path.basename(f)}...")
    try:
        # Read in chunks for memory efficiency
        chunks = pd.read_csv(f, usecols=[1, 5], names=['price', 'timestamp'], header=None, chunksize=2000000)
        for chunk in chunks:
            chunk_start = chunk['timestamp'].iloc[0]
            chunk_end = chunk['timestamp'].iloc[-1]
            
            if chunk_end < start_us: continue
            if chunk_start > end_us: break
                
            mask = (chunk['timestamp'] >= start_us) & (chunk['timestamp'] <= end_us)
            if mask.any():
                filtered = chunk[mask].copy()
                filtered['timestamp'] = pd.to_datetime(filtered['timestamp'], unit='us')
                df_list.append(filtered)
    except Exception as e:
        print(f"Error reading {f}: {e}")
        
    if not df_list:
        return pd.DataFrame(columns=['price', 'timestamp'])
    return pd.concat(df_list)

def run_simulation(config):
    # Parse dates
    start_date = datetime.strptime(config["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(config["end_date"], "%Y-%m-%d")
    
    # Calculate initial UTC start based on config offset, hour, and minute
    start_utc = start_date - timedelta(hours=config["wib_offset"]) + timedelta(hours=config["setup_hour"], minutes=config["setup_minute"])
    end_utc = end_date + timedelta(days=1) # include full last day
    
    # Collect set of required Year-Month strings
    required_months = set()
    current = start_utc
    while current <= end_utc:
        required_months.add(current.strftime("%Y-%m"))
        # Move to first of next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
            
    # Collect only relevant files
    all_files = glob.glob(os.path.join(config["data_path"], "ETHUSDT-aggTrades-*.csv"))
    files = []
    for f in sorted(all_files):
        # Extract YYYY-MM from filename: ETHUSDT-aggTrades-YYYY-MM.csv
        basename = os.path.basename(f)
        parts = basename.replace(".csv", "").split("-")
        if len(parts) >= 3:
            ym = f"{parts[-2]}-{parts[-1]}"
            if ym in required_months:
                files.append(f)
    
    if not files:
        print(f"No data files found for the required period: {required_months}")
        return [], [config["initial_balance"]], {}
        
    equity = config["initial_balance"]
    balance_history = [equity]
    trade_logs = []
    trade_visuals = {}
    
    # Memory Management: Sliding Buffer
    df_buffer = pd.DataFrame()
    file_idx = 0
    current_day = start_date
    
    while current_day <= end_date:
        # Calculate daily target candle start in UTC
        utc_time_target = current_day - timedelta(hours=config["wib_offset"]) + timedelta(hours=config["setup_hour"], minutes=config["setup_minute"])
        utc_pre_candle = utc_time_target - timedelta(minutes=config["timeframe_minutes"])
        day_end = utc_time_target + timedelta(days=1)
        
        # 1. Fill Memory Buffer up to the end of the current trading day
        while file_idx < len(files):
            if not df_buffer.empty and df_buffer['timestamp'].iloc[-1] >= day_end:
                break # We have enough data for this day
                
            f = files[file_idx]
            new_df = load_file_data(f, start_utc, end_utc)
            if not new_df.empty:
                df_buffer = pd.concat([df_buffer, new_df])
            file_idx += 1
            
        if df_buffer.empty:
            print(f"[{current_day.date()}] No data available. Skipping.")
            current_day += timedelta(days=1)
            balance_history.append(equity)
            continue
            
        # Get target candle OHLC based on timeframe config
        utc_time_end_candle = utc_time_target + timedelta(minutes=config["timeframe_minutes"])
        mask_candle = (df_buffer['timestamp'] >= utc_time_target) & (df_buffer['timestamp'] < utc_time_end_candle)
        candle_data = df_buffer[mask_candle]
        
        setup_time_str = f"{config['setup_hour']:02d}:{config['setup_minute']:02d}"
        
        if len(candle_data) < 2:
            print(f"[{current_day.date()}] Missing data for {setup_time_str} candle. Skipping.")
            
            # Prune buffer to prevent memory exhaustion
            df_buffer = df_buffer[df_buffer['timestamp'] >= (day_end - timedelta(minutes=15))]
            current_day += timedelta(days=1)
            balance_history.append(equity)
            continue
            
        high = candle_data['price'].max()
        low = candle_data['price'].min()
        range_val = high - low
        
        # Validation filter
        if range_val < config["min_range_pips"]:
            print(f"[{current_day.date()}] Range {range_val:.2f} < {config['min_range_pips']}. Skipping.")
        else:
            # Valid day, calculate lot
            risk_amount = equity * config["risk_percent"]
            lot_size = max(0.01, risk_amount / range_val) # Clamp minimum lot to 0.01
            
            print(f"\n--- Trading at {current_day.date()} ---")
            print(f"Buy Stop  at {high:.2f}")
            print(f"Sell Stop at {low:.2f}")
            print(f"Range Pips   {range_val:.2f}")
            print(f"Initial Equity: ${equity:.2f}")
            print(f"Risk Amount: ${risk_amount:.2f} (10%)")
            print(f"Lot Size: {lot_size:.4f}")
            
            # Tick data for the rest of the day
            mask_ticks = (df_buffer['timestamp'] >= utc_time_end_candle) & (df_buffer['timestamp'] < day_end)
            tick_subset = df_buffer[mask_ticks]
            tick_prices = tick_subset['price'].values
            tick_times = tick_subset['timestamp'].values
            
            if len(tick_prices) > 0:
                res = simulate_ticks(
                    tick_prices, high, low, lot_size, range_val, low, high, 
                    config["spread_pips"], equity, config["leverage"]
                )
                profit, num_trades, status, t1_ent, t1_ex, t1_p, t1_mdd, t1_margin, t2_ent, t2_ex, t2_p, t2_mdd, t2_margin = res
                
                # Store starting equity before adding trade profits so we can calculate % DD accurately
                starting_equity = equity 
                equity += profit
                if equity < 0:
                    equity = 0.0 # Prevent negative equity on display
                
                # Helper for duration
                def get_duration(start_idx, end_idx, times):
                    if start_idx == -1 or end_idx == -1 or start_idx == end_idx: return "00:00:00"
                    diff = pd.Timestamp(times[end_idx]) - pd.Timestamp(times[start_idx])
                    total_sec = int(diff.total_seconds())
                    h = total_sec // 3600
                    m = (total_sec % 3600) // 60
                    s = total_sec % 60
                    return f"{h:02d}:{m:02d}:{s:02d}"

                # Assign Status labels based on MC condition (-2)
                t1_status_str = 'Margin Call' if (status == -2 and num_trades == 1) else ('Profit' if t1_p > 0 else 'Loss')
                t2_status_str = 'Margin Call' if (status == -2 and num_trades == 2) else ('Profit' if t2_p > 0 else 'Loss')

                # Trade 1
                if t1_ent != -1:
                    e_time = pd.Timestamp(tick_times[t1_ent])
                    eq_t1 = max(starting_equity, 1e-6) # Prevent division by zero
                    trade_logs.append({
                        'Date Open': e_time.strftime('%Y-%m-%d'),
                        'Time Entry': e_time.strftime('%H:%M:%S'),
                        'Time Trading': get_duration(t1_ent, t1_ex, tick_times),
                        'lot': lot_size,
                        'Margin Used': t1_margin,
                        'profit': t1_p,
                        'status': t1_status_str,
                        'Max Drawdown (%)': (t1_mdd / eq_t1) * 100 if t1_status_str != 'Margin Call' else 0.0,
                        'range': range_val,
                        'setup_high': high,
                        'setup_low': low,
                        'equity': equity - (t2_p if t2_ent != -1 else 0),
                        'date': current_day.date() # keep for internal use
                    })
                
                # Trade 2 (Reversal)
                if t2_ent != -1:
                    e_time_2 = pd.Timestamp(tick_times[t2_ent])
                    eq_t2 = max(starting_equity + t1_p, 1e-6) # Start equity for T2 includes T1's closed profit/loss
                    trade_logs.append({
                        'Date Open': e_time_2.strftime('%Y-%m-%d'),
                        'Time Entry': e_time_2.strftime('%H:%M:%S'),
                        'Time Trading': get_duration(t2_ent, t2_ex, tick_times),
                        'lot': lot_size * 2.0,
                        'Margin Used': t2_margin,
                        'profit': t2_p,
                        'status': t2_status_str,
                        'Max Drawdown (%)': (t2_mdd / eq_t2) * 100 if t2_status_str != 'Margin Call' else 0.0,
                        'range': range_val,
                        'setup_high': high,
                        'setup_low': low,
                        'equity': equity,
                        'date': current_day.date()
                    })
                
                # Save a 12-hour downsampled visual slice if a trade occurred (lightweight mapping)
                if num_trades > 0:
                    visual_end = utc_time_target + timedelta(hours=12)
                    plot_df = df_buffer[(df_buffer['timestamp'] >= utc_pre_candle) & (df_buffer['timestamp'] <= visual_end)]
                    if not plot_df.empty:
                        # Resample to 1-minute intervals to save a MASSIVE amount of memory
                        plot_df = plot_df.set_index('timestamp').resample('1min').last().ffill().reset_index()
                        trade_visuals[current_day.date()] = plot_df
                    
                if status == -2:
                    print(f"Result: Margin Call | Trades: {num_trades} | Day Profit: ${profit:+.2f} | End Equity: ${equity:.2f}")
                    print("!!! MARGIN CALL REACHED (Insufficient Balance OR Equity <= 0). STOPPING BACKTEST !!!")
                    current_day += timedelta(days=1)
                    balance_history.append(equity)
                    break
                else:
                    print(f"Result: {status} | Trades: {num_trades} | Day Profit: ${profit:+.2f} | End Equity: ${equity:.2f}")

            else:
                # Calculate what time the candle ends locally for logging purposes
                end_hour = (config["setup_hour"] + (config["setup_minute"] + config["timeframe_minutes"]) // 60) % 24
                end_minute = (config["setup_minute"] + config["timeframe_minutes"]) % 60
                print(f"No tick data after {end_hour:02d}:{end_minute:02d} candle.")
        
        # CRITICAL MEMORY MANAGEMENT: Drop all data before 15 minutes prior to tomorrow's setup
        df_buffer = df_buffer[df_buffer['timestamp'] >= (day_end - timedelta(minutes=15))]
        
        current_day += timedelta(days=1)
        balance_history.append(equity)
        
    return trade_logs, balance_history, trade_visuals

def visualize_single_trade(plot_df, log_entry, config):
    """
    Visualizes a single day of trading using the downsampled visual slice
    """
    date = log_entry['date']
    high = log_entry['setup_high']
    low = log_entry['setup_low']
    range_val = log_entry['range']
    
    utc_setup_start = datetime.combine(date, datetime.min.time()) - timedelta(hours=config["wib_offset"]) + timedelta(hours=config["setup_hour"], minutes=config["setup_minute"])
    utc_setup_end = utc_setup_start + timedelta(minutes=config["timeframe_minutes"])
    
    if plot_df.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df['timestamp'], plot_df['price'], color='#2c3e50', alpha=0.5, linewidth=0.7, label='Price')
    
    # Highlight setup window
    setup_time_str = f"{config['setup_hour']:02d}:{config['setup_minute']:02d}"
    plt.axvspan(utc_setup_start, utc_setup_end, color='yellow', alpha=0.15, label=f'{setup_time_str} Setup')
    
    # Setup levels
    plt.axhline(y=high, color='green', linestyle='--', alpha=0.4, label=f'High: {high:.2f}')
    plt.axhline(y=low, color='red', linestyle='--', alpha=0.4, label=f'Low: {low:.2f}')
    
    plt.title(f"Trade Visualization: {date} | Range: {range_val:.2f} | Result: {log_entry['status']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def generate_mt4_report(logs, history, config):
    """
    Generates a MetaTrader-style HTML report
    """
    df = pd.DataFrame(logs)
    total_trades = len(df)
    wins = len(df[df['profit'] > 0])
    losses = len(df[df['profit'] <= 0])
    net_profit = history[-1] - history[0]
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    total_mdd = df['Max Drawdown (%)'].max()
    
    # Check if we hit a margin call anywhere in the logs
    hit_margin_call = 'Margin Call' in df['status'].values
    
    setup_time_str = f"{config['setup_hour']:02d}:{config['setup_minute']:02d}"
    
    html = f"""
    <html>
    <head>
        <title>Strategy Tester Report: ETHUSD</title>
        <style>
            body {{ font-family: Tahoma, Arial, sans-serif; font-size: 10pt; background-color: #f9f9f9; padding: 20px; }}
            .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); max-width: 1200px; margin: auto; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 9pt; }}
            th {{ background-color: #f2f2f2; }}
            .header {{ font-size: 16pt; font-weight: bold; margin-bottom: 15px; color: #333; border-bottom: 2px solid #ccc; padding-bottom: 5px; }}
            .summary {{ background-color: #fafafa; font-weight: bold; }}
            .mc-banner {{ 
                background-color: #ffebee; 
                color: #c62828; 
                border: 2px solid #c62828; 
                padding: 15px; 
                text-align: center; 
                font-size: 14pt; 
                font-weight: bold; 
                margin-bottom: 20px; 
                border-radius: 5px; 
            }}
            .mc-row {{ background-color: #ffcdd2; font-weight: bold; color: #b71c1c; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">Strategy Tester Report: ETHUSD_Strategy</div>
            {'<div class="mc-banner">⚠️ WARNING: BACKTEST TERMINATED DUE TO MARGIN CALL ⚠️</div>' if hit_margin_call else ''}
            
            <table>
                <tr><th>Symbol</th><td>ETHUSD</td><th>Period</th><td>{config['timeframe_minutes']} Minutes ({setup_time_str} Local)</td></tr>
                <tr><th>Start Date</th><td>{config["start_date"]}</td><th>End Date</th><td>{config["end_date"]}</td></tr>
                <tr><th>Initial Deposit</th><td>{config["initial_balance"]:.2f}</td><th>Spread</th><td>{config["spread_pips"]}</td></tr>
                <tr class="summary"><th>Total Net Profit</th><td style="color: {'green' if net_profit > 0 else 'red'}">{net_profit:.2f}</td><th>Win Rate</th><td>{win_rate:.2f}%</td></tr>
                <tr><th>Total Trades</th><td>{total_trades}</td><th>Max Trade Drawdown</th><td>{total_mdd:.2f}%</td></tr>
            </table>
            
            <h3>Individual Trade History</h3>
            <table>
                <tr>
                    <th>Date Open</th>
                    <th>Time Entry</th>
                    <th>Time Trading</th>
                    <th>Lot Size</th>
                    <th>Margin Used ($)</th>
                    <th>Max DD (%)</th>
                    <th>Profit ($)</th>
                    <th>Status</th>
                    <th>Equity ($)</th>
                </tr>
    """
    
    for _, row in df.iterrows():
        is_mc = row['status'] == 'Margin Call'
        row_class = ' class="mc-row"' if is_mc else ''
        status_color = 'inherit' if is_mc else ('green' if row['status'] == 'Profit' else 'red')
        profit_color = 'inherit' if is_mc else ('green' if row['profit'] > 0 else 'red')
        
        html += f"""
                <tr{row_class}>
                    <td>{row['Date Open']}</td>
                    <td>{row['Time Entry']}</td>
                    <td>{row['Time Trading']}</td>
                    <td>{row['lot']:.4f}</td>
                    <td>{row['Margin Used']:.2f}</td>
                    <td style="color: {'inherit' if is_mc else 'red'};">{row['Max Drawdown (%)']:.2f}%</td>
                    <td style="color: {profit_color};">{row['profit']:.2f}</td>
                    <td style="color: {status_color};">{row['status']}</td>
                    <td>{row['equity']:.2f}</td>
                </tr>
        """
        
    html += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open("backtest_report.html", "w") as f:
        f.write(html)
    print("\nMT4-style report generated: backtest_report.html")

if __name__ == "__main__":
    print(f"Starting Backtest from {CONFIG['start_date']} to {CONFIG['end_date']}...")
    logs, history, trade_visuals = run_simulation(CONFIG)
    
    # Print summary
    print("\n--- Backtest Summary ---")
    if logs:
        df_logs = pd.DataFrame(logs)
        
        # --- Calculate and Print Summary Statistics ---
        total_trades = len(df_logs)
        winning_trades = len(df_logs[df_logs['profit'] > 0])
        losing_trades = len(df_logs[df_logs['profit'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
        
        print(f"Total Trades   : {total_trades}")
        print(f"Winning Trades : {winning_trades}")
        print(f"Losing Trades  : {losing_trades}")
        print(f"Win Rate       : {win_rate:.2f}%\n")
        
        # Select and reorder columns for clean terminal display
        summary_cols = ['Date Open', 'Time Entry', 'Time Trading', 'lot', 'Margin Used', 'profit', 'Max Drawdown (%)', 'status', 'equity']
        
        # Format the percentage and money columns specifically for terminal printing
        df_display = df_logs[summary_cols].copy()
        df_display['Max Drawdown (%)'] = df_display['Max Drawdown (%)'].apply(lambda x: f"{x:.2f}%")
        df_display['Margin Used'] = df_display['Margin Used'].apply(lambda x: f"${x:.2f}")
        print(df_display)
        
        generate_mt4_report(logs, history, CONFIG)
        
        # Plot Equity Curve
        plt.figure("Equity Curve", figsize=(10, 5))
        plt.plot(history, marker='o', color='blue', label='Equity Curve')
        plt.title(f"Equity Curve (Initial: ${CONFIG['initial_balance']})")
        plt.xlabel("Days")
        plt.ylabel("Balance ($)")
        plt.grid(True)
        plt.savefig("equity_curve.png")
        print("Equity curve saved: equity_curve.png")
        
        # Visualize ALL days with trades (unique dates)
        seen_dates = set()
        trade_days = [log for log in logs if log['date'] not in seen_dates and not seen_dates.add(log['date'])]
        print(f"\nLaunching visualization for {len(trade_days)} traded days...")
        for log in trade_days:
            if log['date'] in trade_visuals:
                visualize_single_trade(trade_visuals[log['date']], log, CONFIG)
            
    else:
        print("No trades executed.")
        
    print(f"\nFinal Balance: ${history[-1]:.2f}")
    # plt.show() # Shows all generated windows