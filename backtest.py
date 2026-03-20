import pandas as pd
import numpy as np
from numba import njit
import os
import glob
from datetime import datetime, timedelta
import multiprocessing as mp
import matplotlib.pyplot as plt

# Parameters from user
INITIAL_BALANCE = 100.0
LEVERAGE = 400
RISK_PERCENT = 0.10
MIN_RANGE = 8.0
SPREAD = 2.0  # Default spread as requested
WIB_OFFSET = 7 # WIB is UTC+7

@njit
def simulate_ticks(ticks, buy_stop, sell_stop, lot_size_in, tp_distance, sl_price_buy, sl_price_sell, spread):
    """
    Simultaneously watches 2 stop orders.
    Returns: (total_profit, num_trades, status, t1_entry, t1_exit, t1_p, t1_mdd, t2_entry, t2_exit, t2_p, t2_mdd)
    status: 1 (Win), -1 (Loss), 0 (No trade)
    """
    pos_type = 0 # 0: none, 1: long, 2: short
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    lot_size = lot_size_in
    
    trades_count = 0
    total_profit = 0.0
    
    t1_entry, t1_exit, t1_p, t1_mdd = -1, -1, 0.0, 0.0
    t2_entry, t2_exit, t2_p, t2_mdd = -1, -1, 0.0, 0.0
    
    buy_pending = True
    sell_pending = True
    
    # Spread adjustments
    # Long: Entry = Trigger + Spread/2, Exit = Trigger - Spread/2
    # Short: Entry = Trigger - Spread/2, Exit = Trigger + Spread/2
    s2 = spread / 2.0
    
    for i in range(len(ticks)):
        p = ticks[i]
        
        if pos_type == 0:
            if buy_pending and p >= buy_stop:
                pos_type = 1
                entry_price = buy_stop + s2
                tp_price = buy_stop + tp_distance # Target is relative to original stop
                sl_price = sl_price_buy - s2
                t1_entry, trades_count = i, 1
                buy_pending = False
            elif sell_pending and p <= sell_stop:
                pos_type = 2
                entry_price = sell_stop - s2
                tp_price = sell_stop - tp_distance
                sl_price = sl_price_sell + s2
                t1_entry, trades_count = i, 1
                sell_pending = False
                
        elif pos_type == 1: # Long
            # Track unrealized drawdown
            unrealized_p = (p - entry_price) * lot_size
            if unrealized_p < 0:
                dd = abs(unrealized_p)
                if trades_count == 1: t1_mdd = max(t1_mdd, dd)
                else: t2_mdd = max(t2_mdd, dd)
                
            if p >= (tp_price + s2): # Exit TP
                res_p = (tp_price - entry_price) * lot_size
                if trades_count == 1:
                    t1_p, t1_exit = res_p, i
                    return t1_p, 1, 1, t1_entry, t1_exit, t1_p, t1_mdd, -1, -1, 0.0, 0.0
                else:
                    t2_p, t2_exit = res_p, i
                    return t1_p + t2_p, 2, 1, t1_entry, t1_exit, t1_p, t1_mdd, t2_entry, t2_exit, t2_p, t2_mdd
            elif p <= sl_price: # Exit SL
                res_p = (sl_price - entry_price) * lot_size
                if trades_count == 1:
                    t1_p, t1_exit = res_p, i
                    # Reverse to Short
                    pos_type, entry_price = 2, sl_price - s2
                    lot_size *= 2.0
                    tp_price, sl_price = sl_price - tp_distance, buy_stop + s2
                    t2_entry, trades_count = i, 2
                else:
                    t2_p, t2_exit = res_p, i
                    return t1_p + t2_p, 2, -1, t1_entry, t1_exit, t1_p, t1_mdd, t2_entry, t2_exit, t2_p, t2_mdd
                    
        elif pos_type == 2: # Short
            # Track unrealized drawdown
            unrealized_p = (entry_price - p) * lot_size
            if unrealized_p < 0:
                dd = abs(unrealized_p)
                if trades_count == 1: t1_mdd = max(t1_mdd, dd)
                else: t2_mdd = max(t2_mdd, dd)

            if p <= (tp_price - s2): # Exit TP
                res_p = (entry_price - tp_price) * lot_size
                if trades_count == 1:
                    t1_p, t1_exit = res_p, i
                    return t1_p, 1, 1, t1_entry, t1_exit, t1_p, t1_mdd, -1, -1, 0.0, 0.0
                else:
                    t2_p, t2_exit = res_p, i
                    return t1_p + t2_p, 2, 1, t1_entry, t1_exit, t1_p, t1_mdd, t2_entry, t2_exit, t2_p, t2_mdd
            elif p >= sl_price: # Exit SL
                res_p = (entry_price - sl_price) * lot_size
                if trades_count == 1:
                    t1_p, t1_exit = res_p, i
                    # Reverse to Long
                    pos_type, entry_price = 1, sl_price + s2
                    lot_size *= 2.0
                    tp_price, sl_price = sl_price + tp_distance, sell_stop - s2
                    t2_entry, trades_count = i, 2
                else:
                    t2_p, t2_exit = res_p, i
                    return t1_p + t2_p, 2, -1, t1_entry, t1_exit, t1_p, t1_mdd, t2_entry, t2_exit, t2_p, t2_mdd
                    
    return total_profit, trades_count, 0, t1_entry, t1_exit, t1_p, t1_mdd, t2_entry, t2_exit, t2_p, t2_mdd

def _process_single_file(args):
    """Worker function for multiprocessing"""
    f, start_utc, end_utc = args
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
        return None
    return pd.concat(df_list)

def load_filtered_data_parallel(files, start_utc, end_utc):
    """
    Load multiple files in parallel using Multiprocessing
    """
    args_list = [(f, start_utc, end_utc) for f in files]
    
    # Use number of CPU cores or number of files, whichever is smaller
    num_workers = min(mp.cpu_count(), len(files))
    if num_workers < 1: num_workers = 1
    
    print(f"Starting parallel load with {num_workers} workers...")
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_process_single_file, args_list)
    
    # Combine results
    valid_dfs = [r for r in results if r is not None]
    if not valid_dfs:
        return pd.DataFrame(columns=['price', 'timestamp'])
        
    print("Combining data and sorting...")
    return pd.concat(valid_dfs).sort_values('timestamp')

def run_simulation(data_path, start_date_str, end_date_str):
    # Parse dates
    start_date = datetime.strptime(start_date_str, "%d/%m/%Y")
    end_date = datetime.strptime(end_date_str, "%d/%m/%Y")
    
    # We need Dec 31st 22:15 UTC for Jan 1st 05:15 WIB
    start_utc = start_date - timedelta(hours=WIB_OFFSET) + timedelta(hours=5, minutes=15)
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
    all_files = glob.glob(os.path.join(data_path, "ETHUSDT-aggTrades-*.csv"))
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
        return [], [INITIAL_BALANCE], None
        
    df_full = load_filtered_data_parallel(files, start_utc, end_utc)
    
    equity = INITIAL_BALANCE
    balance_history = [equity]
    trade_logs = []
    
    current_day = start_date
    while current_day <= end_date:
        # 05:15 WIB = 22:15 UTC (previous day)
        utc_time_target = current_day - timedelta(hours=WIB_OFFSET) + timedelta(hours=5, minutes=15)
        utc_time_end_candle = utc_time_target + timedelta(minutes=15)
        
        # Get 05:15-05:30 WIB candle OHLC
        mask_candle = (df_full['timestamp'] >= utc_time_target) & (df_full['timestamp'] < utc_time_end_candle)
        candle_data = df_full[mask_candle]
        
        if len(candle_data) < 2:
            print(f"[{current_day.date()}] Missing data for 05:15 WIB candle. Skipping.")
            current_day += timedelta(days=1)
            continue
            
        high = candle_data['price'].max()
        low = candle_data['price'].min()
        range_val = high - low
        
        # Validation filter
        if range_val < MIN_RANGE:
            print(f"[{current_day.date()}] Range {range_val:.2f} < {MIN_RANGE}. Skipping.")
        else:
            # Valid day, calculate lot
            risk_amount = equity * RISK_PERCENT
            lot_size = risk_amount / range_val
            
            print(f"\n--- Trading at {current_day.date()} ---")
            print(f"Buy Stop  at {high:.2f}")
            print(f"Sell Stop at {low:.2f}")
            print(f"Range Pips   {range_val:.2f}")
            print(f"Initial Equity: ${equity:.2f}")
            print(f"Risk Amount: ${risk_amount:.2f} (10%)")
            print(f"Lot Size: {lot_size:.4f}")
            
            # Tick data for the rest of the day (until next day 05:15 WIB)
            day_end = utc_time_target + timedelta(days=1)
            mask_ticks = (df_full['timestamp'] >= utc_time_end_candle) & (df_full['timestamp'] < day_end)
            tick_subset = df_full[mask_ticks]
            tick_prices = tick_subset['price'].values
            tick_times = tick_subset['timestamp'].values
            
            if len(tick_prices) > 0:
                res = simulate_ticks(tick_prices, high, low, lot_size, range_val, low, high, SPREAD)
                profit, num_trades, status, t1_ent, t1_ex, t1_p, t1_mdd, t2_ent, t2_ex, t2_p, t2_mdd = res
                equity += profit
                
                # Helper for duration
                def get_duration(start_idx, end_idx, times):
                    if start_idx == -1 or end_idx == -1: return "N/A"
                    diff = times[end_idx] - times[start_idx]
                    total_sec = int(diff.total_seconds())
                    h = total_sec // 3600
                    m = (total_sec % 3600) // 60
                    s = total_sec % 60
                    return f"{h:02d}:{m:02d}:{s:02d}"

                # Trade 1
                if t1_ent != -1:
                    e_time = tick_times[t1_ent]
                    trade_logs.append({
                        'Date Open': e_time.strftime('%Y-%m-%d'),
                        'Time Entry': e_time.strftime('%H:%M:%S'),
                        'Time Trading': get_duration(t1_ent, t1_ex, tick_times),
                        'lot': lot_size,
                        'profit': t1_p,
                        'status': 'Profit' if t1_p > 0 else 'Loss',
                        'Max Drawdown': t1_mdd,
                        'range': range_val,
                        'setup_high': high,
                        'setup_low': low,
                        'equity': equity - (t2_p if t2_ent != -1 else 0),
                        'date': current_day.date() # keep for internal use
                    })
                
                # Trade 2 (Reversal)
                if t2_ent != -1:
                    e_time_2 = tick_times[t2_ent]
                    trade_logs.append({
                        'Date Open': e_time_2.strftime('%Y-%m-%d'),
                        'Time Entry': e_time_2.strftime('%H:%M:%S'),
                        'Time Trading': get_duration(t2_ent, t2_ex, tick_times),
                        'lot': lot_size * 2.0,
                        'profit': t2_p,
                        'status': 'Profit' if t2_p > 0 else 'Loss',
                        'Max Drawdown': t2_mdd,
                        'range': range_val,
                        'setup_high': high,
                        'setup_low': low,
                        'equity': equity,
                        'date': current_day.date()
                    })
                    
                print(f"Result: {status} | Trades: {num_trades} | Day Profit: ${profit:+.2f} | End Equity: ${equity:.2f}")
            else:
                print(f"No tick data after 05:30 WIB candle.")
        
        current_day += timedelta(days=1)
        balance_history.append(equity)
        
    return trade_logs, balance_history, df_full

def visualize_single_trade(df_full, log_entry):
    """
    Visualizes a single day of trading using the already loaded data
    """
    date = log_entry['date']
    high = log_entry['setup_high']
    low = log_entry['setup_low']
    range_val = log_entry['range']
    
    # 05:15 WIB = 22:15 UTC (previous day)
    utc_setup_start = datetime.combine(date, datetime.min.time()) - timedelta(hours=WIB_OFFSET) + timedelta(hours=5, minutes=15)
    utc_setup_end = utc_setup_start + timedelta(minutes=15)
    utc_pre_candle = utc_setup_start - timedelta(minutes=15)
    
    # Filter data for visualization (Show 12 hours after setup)
    day_start = utc_pre_candle
    day_end = utc_setup_start + timedelta(hours=12) 
    
    plot_df = df_full[(df_full['timestamp'] >= day_start) & (df_full['timestamp'] <= day_end)]
    
    if plot_df.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df['timestamp'], plot_df['price'], color='#2c3e50', alpha=0.5, linewidth=0.7, label='Price')
    
    # Highlight setup window
    plt.axvspan(utc_setup_start, utc_setup_end, color='yellow', alpha=0.15, label='05:15 Setup')
    
    # Setup levels
    plt.axhline(y=high, color='green', linestyle='--', alpha=0.4, label=f'High: {high:.2f}')
    plt.axhline(y=low, color='red', linestyle='--', alpha=0.4, label=f'Low: {low:.2f}')
    
    plt.title(f"Trade Visualization: {date} | Range: {range_val:.2f} | Result: {log_entry['status']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def generate_mt4_report(logs, history, start_date, end_date):
    """
    Generates a MetaTrader-style HTML report
    """
    df = pd.DataFrame(logs)
    total_trades = len(df)
    wins = len(df[df['status'] == 'Profit'])
    losses = len(df[df['status'] == 'Loss'])
    net_profit = history[-1] - history[0]
    win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
    total_mdd = df['Max Drawdown'].max()
    
    html = f"""
    <html>
    <head>
        <title>Strategy Tester Report: ETHUSD</title>
        <style>
            body {{ font-family: Tahoma, Arial, sans-serif; font-size: 10pt; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ccc; padding: 5px; text-align: left; font-size: 9pt; }}
            th {{ background-color: #f2f2f2; }}
            .header {{ font-size: 14pt; font-weight: bold; margin-bottom: 10px; }}
            .summary {{ background-color: #fafafa; }}
        </style>
    </head>
    <body>
        <div class="header">Strategy Tester Report: ETHUSD_Strategy</div>
        <table>
            <tr><th>Symbol</th><td>ETHUSD</td><th>Period</th><td>15 Minutes (05:15 WIB)</td></tr>
            <tr><th>Start Date</th><td>{start_date}</td><th>End Date</th><td>{end_date}</td></tr>
            <tr><th>Initial Deposit</th><td>{INITIAL_BALANCE:.2f}</td><th>Spread</th><td>{SPREAD}</td></tr>
            <tr class="summary"><th>Total Net Profit</th><td>{net_profit:.2f}</td><th>Win Rate</th><td>{win_rate:.2f}%</td></tr>
            <tr><th>Total Trades</th><td>{total_trades}</td><th>Max Trade Drawdown</th><td>${total_mdd:.2f}</td></tr>
        </table>
        
        <h3>Individual Trade History</h3>
        <table>
            <tr>
                <th>Date Open</th>
                <th>Time Entry</th>
                <th>Time Trading</th>
                <th>Lot Size</th>
                <th>Max DD ($)</th>
                <th>Profit ($)</th>
                <th>Status</th>
                <th>Equity ($)</th>
            </tr>
    """
    
    for _, row in df.iterrows():
        html += f"""
            <tr>
                <td>{row['Date Open']}</td>
                <td>{row['Time Entry']}</td>
                <td>{row['Time Trading']}</td>
                <td>{row['lot']:.4f}</td>
                <td style="color: red;">{row['Max Drawdown']:.2f}</td>
                <td style="color: {'green' if row['profit'] > 0 else 'red'}">{row['profit']:.2f}</td>
                <td>{row['status']}</td>
                <td>{row['equity']:.2f}</td>
            </tr>
        """
        
    html += """
        </table>
    </body>
    </html>
    """
    
    with open("backtest_report.html", "w") as f:
        f.write(html)
    print("\nMT4-style report generated: backtest_report.html")

if __name__ == "__main__":
    data_path = r"d:\repos\ethusd\data\raw"
    # User's dates
    start = "10/02/2026"
    end = "20/02/2026"
    
    print(f"Starting Backtest from {start} to {end}...")
    logs, history, df_full = run_simulation(data_path, start, end)
    
    # Print summary
    print("\n--- Backtest Summary ---")
    if logs:
        df_logs = pd.DataFrame(logs)
        # Select and reorder columns for clean terminal display
        summary_cols = ['Date Open', 'Time Entry', 'Time Trading', 'lot', 'profit', 'Max Drawdown', 'status', 'equity']
        print(df_logs[summary_cols])
        generate_mt4_report(logs, history, start, end)
        
        # Plot Equity Curve
        plt.figure("Equity Curve", figsize=(10, 5))
        plt.plot(history, marker='o', color='blue', label='Equity Curve')
        plt.title(f"Equity Curve (Initial: ${INITIAL_BALANCE})")
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
            visualize_single_trade(df_full, log)
            
    else:
        print("No trades executed.")
        
    print(f"\nFinal Balance: ${history[-1]:.2f}")
    plt.show() # Shows all generated windows
