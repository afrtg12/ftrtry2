from logging import log
import os
import yfinance as yf
from alpaca_trade_api.rest import REST
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# Load Alpaca API keys from environment variables (GitHub Secrets)
API_KEY_SCALPING = os.getenv('ALPACA_API_KEY_1')
SECRET_KEY_SCALPING = os.getenv('ALPACA_SECRET_KEY_1')
API_KEY_BOLLINGER = os.getenv('ALPACA_API_KEY_2')
SECRET_KEY_BOLLINGER = os.getenv('ALPACA_SECRET_KEY_2')
BASE_URL = 'https://paper-api.alpaca.markets'

# Connect to Alpaca Paper Trading Accounts
client_scalping = REST(API_KEY_SCALPING, SECRET_KEY_SCALPING, BASE_URL)
client_bollinger = REST(API_KEY_BOLLINGER, SECRET_KEY_BOLLINGER, BASE_URL)

# Symbols to trade
symbols = ['NFLX', 'TSLA', 'AMD','INTC']

# Store last trade times to enforce cooldown
last_trade_times_scalping = {}
last_trade_times_bollinger = {}

# Fetch the latest 5-minute interval data using yfinance
def fetch_data(symbol):
    df = yf.download(tickers=symbol, period="5d", interval="5m")
    return df

# Calculate RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# Calculate ATR
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    # Use np.maximum for element-wise comparison
    tr = high_low.combine(high_close, np.maximum).combine(low_close, np.maximum) 
    return tr.rolling(window=period).mean()

# Calculate Bollinger Bands
def calculate_bollinger_bands(prices, period=14, std_dev=1.5):
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = ma + (std_dev * std)
    lower_band = ma - (std_dev * std)
    return upper_band, lower_band
# 📄 Log File Path
LOG_FILE = "trade_log.txt"

# ✏️ Logging Function
def log_trade(symbol, strategy, decision, price):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Convert price to a float if it's a Series
    price = price.item() if isinstance(price, pd.Series) else price  
    log_entry = f"{current_time} | Symbol: {symbol} | Strategy: {strategy} | Decision: {decision} | Price: {price:.2f}\n"
    
    with open(LOG_FILE, "a") as file:
        file.write(log_entry)
    
    print(f"Logged trade: {log_entry.strip()}")
# 🔥 Updated Order Execution with Logging
def execute_trade(symbol, side, position_size, client, stop_loss_price=None, take_profit_price=None, strategy_name="Unknown"):
    """
    Executes a buy/sell order on Alpaca with optional stop-loss and take-profit, and logs the trade.
    """
    try:
        client.submit_order(
            symbol=symbol,
            qty=position_size,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        print(f'{side.upper()} ORDER PLACED: {position_size} shares of {symbol}')
        
        # 📝 Log the trade
        log_trade(symbol, strategy_name, side.upper(), client.get_last_trade(symbol).price)

        # Set Stop-Loss
        if stop_loss_price:
            client.submit_order(
                symbol=symbol,
                qty=position_size,
                side='sell' if side == 'buy' else 'buy',
                type='stop',
                stop_price=stop_loss_price,
                time_in_force='gtc'
            )
            print(f'STOP-LOSS set at {stop_loss_price} for {symbol}')

        # Set Take-Profit
        if take_profit_price:
            client.submit_order(
                symbol=symbol,
                qty=position_size,
                side='sell' if side == 'buy' else 'buy',
                type='limit',
                limit_price=take_profit_price,
                time_in_force='gtc'
            )
            print(f'TAKE-PROFIT set at {take_profit_price} for {symbol}')

    except Exception as e:
        print(f'Error placing {side.upper()} order for {symbol}: {e}')

# --- Optimized Contrarian Scalping Strategy ---
def contrarian_scalping(symbol, client, cooldown_period=12, atr_period=14, take_profit_factor=2.5, stop_loss_factor=1.5):
    df = fetch_data(symbol)
    if df.empty:
        print(f"No data for {symbol}")
        return

    close_prices = df['Close']
    rsi = calculate_rsi(close_prices).iloc[-1]
    atr = calculate_atr(df, period=atr_period).iloc[-1]
    latest_price = close_prices.iloc[-1]

    # Enforce Cooldown
    current_time = datetime.now()
    last_trade_time = last_trade_times_scalping.get(symbol, current_time - timedelta(minutes=cooldown_period + 1))

    if (current_time - last_trade_time).total_seconds() < cooldown_period * 60:
        print(f"Cooldown active for {symbol} in Scalping. Skipping trade.")
        return

    # Position size based on ATR
    cash = float(client.get_account().cash)
    risk_per_trade = cash * 0.01
    # ✅ Ensure ATR is a scalar
    atr = atr if np.isscalar(atr) else atr.item()
    position_size = max(int(risk_per_trade / atr), 1)

    # Set Stop-Loss and Take-Profit
    if rsi.item() > 80:
        stop_loss_price = latest_price + (stop_loss_factor * atr)
        take_profit_price = latest_price - (take_profit_factor * atr)
        execute_trade(symbol, 'sell', position_size, client, stop_loss_price, take_profit_price)
        last_trade_times_scalping[symbol] = current_time

    elif rsi.item() < 20:
        stop_loss_price = latest_price - (stop_loss_factor * atr)
        take_profit_price = latest_price + (take_profit_factor * atr)
        execute_trade(symbol, 'buy', position_size, client, stop_loss_price, take_profit_price)
        last_trade_times_scalping[symbol] = current_time
    else:
        log_trade(symbol, "ContrarianScalping", "No Trade", latest_price)
        print(f"No trade taken for {symbol} based on RSI: {rsi.item()}")


# --- Optimized BollingerROC Strategy ---
def bollinger_roc(symbol, client, cooldown_period=12, atr_period=14, take_profit_factor=2.5, stop_loss_factor=1.5):
    df = fetch_data(symbol)
    if df.empty:
        print(f"No data for {symbol}")
        return

    close_prices = df['Close']
    upper_band, lower_band = calculate_bollinger_bands(close_prices)
    roc = close_prices.pct_change(periods=6).iloc[-1]
    atr = calculate_atr(df, period=atr_period).iloc[-1]
    latest_price = close_prices.iloc[-1]

    # 🔎 Debugging after extraction
    print(f"\nSymbol: {symbol}")
    print(f"Latest Price (Raw): {latest_price}")
    print(f"Lower Band (Raw): {lower_band.iloc[-1]}")
    print(f"Upper Band (Raw): {upper_band.iloc[-1]}")
    print(f"ROC (Raw): {roc}")

    # ✅ Force to Scalar Values
    latest_price = latest_price.values[0] if isinstance(latest_price, pd.Series) else latest_price
    lower_band_value = lower_band.iloc[-1].values[0] if isinstance(lower_band.iloc[-1], pd.Series) else lower_band.iloc[-1]
    upper_band_value = upper_band.iloc[-1].values[0] if isinstance(upper_band.iloc[-1], pd.Series) else upper_band.iloc[-1]
    roc_value = roc.values[0] if isinstance(roc, pd.Series) else roc

    print(f"Latest Price (Scalar): {latest_price}")
    print(f"Lower Band (Scalar): {lower_band_value}")
    print(f"Upper Band (Scalar): {upper_band_value}")
    print(f"ROC (Scalar): {roc_value}")

    # Enforce Cooldown
    current_time = datetime.now()
    last_trade_time = last_trade_times_bollinger.get(symbol, current_time - timedelta(minutes=cooldown_period + 1))

    if (current_time - last_trade_time).total_seconds() < cooldown_period * 60:
        print(f"Cooldown active for {symbol} in BollingerROC. Skipping trade.")
        return

    # Position size based on ATR
    cash = float(client.get_account().cash)
    risk_per_trade = cash * 0.02

    # ✅ Ensure ATR is a scalar
    atr = atr if np.isscalar(atr) else atr.item()

    position_size = max(int(risk_per_trade / atr), 1)

    # ✅ Corrected Strategy Conditions
    if latest_price < lower_band_value and roc_value > 0:
        stop_loss_price = latest_price - (stop_loss_factor * atr)
        take_profit_price = latest_price + (take_profit_factor * atr)
        execute_trade(symbol, 'buy', position_size, client, stop_loss_price, take_profit_price)
        last_trade_times_bollinger[symbol] = current_time

    elif latest_price > upper_band_value and roc_value < 0:
        stop_loss_price = latest_price + (stop_loss_factor * atr)
        take_profit_price = latest_price - (take_profit_factor * atr)
        execute_trade(symbol, 'sell', position_size, client, stop_loss_price, take_profit_price)
        last_trade_times_bollinger[symbol] = current_time
    else:
        log_trade(symbol, "BollingerROC", "No Trade", latest_price)
        print(f"No trade taken for {symbol} based on ROC: {roc_value}")


# 🚀 Run Both Strategies
def run_trading():
    for symbol in symbols:
        contrarian_scalping(symbol, client_scalping)  # Account 1
        bollinger_roc(symbol, client_bollinger)       # Account 2

if __name__ == "__main__":
    run_trading()
