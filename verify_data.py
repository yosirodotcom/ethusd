import pandas as pd
import os

data_dir = r"d:\repos\ethusd\data\raw"
file_path = os.path.join(data_dir, "ETHUSDT-aggTrades-2025-01.csv")

# agg_trade_id, price, quantity, first_trade_id, last_trade_id, timestamp, is_buyer_maker, is_best_match
columns = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 'timestamp', 'is_buyer_maker', 'is_best_match']

df = pd.read_csv(file_path, names=columns, nrows=10)
print("Columns:", df.columns.tolist())
print(df.head())

# Check timestamp unit
ts = df['timestamp'].iloc[0]
print(f"\nExample timestamp: {ts}")

# Try microseconds (16 digits for 2025)
if len(str(ts)) >= 15:
    print("Detected Microseconds")
    print(pd.to_datetime(ts, unit='us'))
else:
    print("Detected Milliseconds")
    print(pd.to_datetime(ts, unit='ms'))
