# Import necessary libraries
import backtrader as bt
import backtrader.analyzers as btanalyzers
import yfinance as yf
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
trade_logs = []
summary_logs = []
# Fetch historical data using yfinance
def fetch_historical_data(symbol, start_date, end_date, interval='25m'):
    df = yf.download(tickers=symbol, start=start_date, end=end_date, interval=interval)
    
    if df.empty:
        raise ValueError(f"No data found for symbol {symbol} between {start_date} and {end_date}.")
    
    df.index.name = 'datetime'

    # Keep only the relevant columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    return df

class BalanceOfPower(bt.Indicator):
    lines = ('bop',)
    plotinfo = dict(subplot=True)

    def __init__(self):
        self.lines.bop = (self.data.close - self.data.open) / (self.data.high - self.data.low + 1e-9)  # Add small epsilon to avoid division by zero


# Define Bollinger Bands with Rate of Change (ROC) Strategy
class BollingerROC(bt.Strategy):
    params = (('period', 14), ('std_dev', 1.5), ('roc_period', 6),)

    def __init__(self):
        self.boll = bt.indicators.BollingerBands(self.data.close, period=self.p.period, devfactor=self.p.std_dev)
        self.roc = bt.indicators.ROC(self.data.close, period=self.p.roc_period)
        self.order = None
        self.buyprice = None
        self.initial_position_opened = False
        self.buycomm = None

    def log_trade(self, decision, successful=None):
        trade_logs.append({
            'strategy': self.__class__.__name__,
            'time': self.data.datetime.datetime(0),
            'decision': decision,
            'portfolio_value': self.broker.getvalue(),
            'successful': successful
        })

    def next(self):
        if not self.initial_position_opened:
            self.order = self.buy(size=10)  # Buy 10 shares on the first bar
            self.log_trade('buy')
            self.initial_position_opened = True
            return

        position_size = 10

        if self.order:
            return

        if not self.position:
            if self.data.close[0] < self.boll.lines.bot[0] and self.roc[0] > 0:
                self.order = self.buy(size=position_size)
                self.log_trade('buy')
        else:
            if self.data.close[0] > self.boll.lines.top[0] and self.roc[0] < 0:
                self.order = self.sell(size=position_size)
                self.log_trade('sell')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                profit = order.executed.price - self.buyprice
                successful = profit > 0
                self.log_trade('sell', successful)
        self.order = None

    def stop(self):
        successful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'])
        unsuccessful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'] is False)
        summary_logs.append({
            'symbol': self.data._name,
            'strategy': self.__class__.__name__,
            'starting_value': self.broker.startingcash,
            'ending_value': self.broker.getvalue(),
            'successful_trades': successful_trades,
            'unsuccessful_trades': unsuccessful_trades
        })
        print(f'BollingerROC Strategy: Successful Trades: {successful_trades}, Unsuccessful Trades: {unsuccessful_trades}')

# Define RSI, Balance of Power (BOP), and MACD Strategy
class RSI_BOP_MACD(bt.Strategy):
    params = (('rsi_period', 7), ('macd_short', 12), ('macd_long', 34), ('macd_signal', 6),)

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.bop = BalanceOfPower(self.data)  # Custom BOP Indicator
        self.macd = bt.indicators.MACD(self.data.close, 
                                       period_me1=self.p.macd_short, 
                                       period_me2=self.p.macd_long, 
                                       period_signal=self.p.macd_signal)
        self.order = None
        self.buyprice = None
        self.initial_position_opened = False
        self.buycomm = None

    def log_trade(self, decision, successful=None):
        trade_logs.append({
            'strategy': self.__class__.__name__,
            'time': self.data.datetime.datetime(0),
            'decision': decision,
            'portfolio_value': self.broker.getvalue(),
            'successful': successful
        })

    def calculate_position_size(self):
        cash = self.broker.get_cash()  # Current available cash
        risk_amount = cash * 0.02  # Risk Amount

        # Stop Loss = 1% of the current price
        stop_loss_price = self.data.close[0] * 0.01

        # Calculate the number of shares to trade
        position_size = risk_amount / stop_loss_price

        return int(position_size)  # Return whole shares

    def next(self):
        if not self.initial_position_opened:
            self.order = self.buy(size=10)  # Buy 10 shares on the first bar
            self.log_trade('buy')
            self.initial_position_opened = True
            return

        position_size = self.calculate_position_size()

        if self.order:
            return

        high_rsi = 70
        high_bop = 0.5
        high_macd = self.macd.macd[0] * 1.25  # Adjust as needed

        if not self.position:
            if (self.rsi[0] > high_rsi and
                self.bop[0] > high_bop and
                self.macd.macd[0] > high_macd):
                self.order = self.sell(size=position_size)
                self.log_trade('sell')
        else:
            if self.rsi[0] < high_rsi:
                self.order = self.buy(size=position_size)
                self.log_trade('buy')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                profit = self.buyprice - order.executed.price
                successful = profit > 0
                self.log_trade('sell', successful)
        self.order = None

    def stop(self):
        successful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'])

        unsuccessful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'] is False)
        summary_logs.append({
            'symbol': self.data._name,
            'strategy': self.__class__.__name__,
            'starting_value': self.broker.startingcash,
            'ending_value': self.broker.getvalue(),
            'successful_trades': successful_trades,
            'unsuccessful_trades': unsuccessful_trades
        })
        print(f'RSI_BOP_MACD Strategy: Successful Trades: {successful_trades}, Unsuccessful Trades: {unsuccessful_trades}')

# Define Moving Average Crossover Strategy
class MovingAverageCrossover(bt.Strategy):
    params = (('short_window', 20), ('long_window', 50),)

    def __init__(self):
        self.short_mavg = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.short_window)
        self.long_mavg = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.long_window)
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def log_trade(self, decision, successful=None):
        trade_logs.append({
            'strategy': self.__class__.__name__,
            'time': self.data.datetime.datetime(0),
            'decision': decision,
            'portfolio_value': self.broker.getvalue(),
            'successful': successful
        })

    def next(self):
        position_size = 10

        if self.order:
            return

        if not self.position:
            if self.short_mavg[0] > self.long_mavg[0]:
                self.order = self.buy(size=position_size)
                self.log_trade('buy')
        else:
            if self.short_mavg[0] < self.long_mavg[0]:
                self.order = self.sell(size=position_size)
                self.log_trade('sell')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                profit = order.executed.price - self.buyprice
                successful = profit > 0
                self.log_trade('sell', successful)
        self.order = None

    def stop(self):
        successful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'])

        unsuccessful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'] is False)
        summary_logs.append({
    'symbol': self.data._name,
    'strategy': self.__class__.__name__,
    'starting_value': self.broker.startingcash,
    'ending_value': self.broker.getvalue(),
    'successful_trades': successful_trades,
    'unsuccessful_trades': unsuccessful_trades
})
        print(f'MovingAverageCrossover Strategy: Successful Trades: {successful_trades}, Unsuccessful Trades: {unsuccessful_trades}')


# Define MACD Crossover Strategy
class MACDCrossover(bt.Strategy):
    params = (
        ('macd1', 12),  # Period for the fast EMA
        ('macd2', 26),  # Period for the slow EMA
        ('macdsig', 9),  # Period for the signal line
    )

    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd1,
            period_me2=self.p.macd2,
            period_signal=self.p.macdsig
        )
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def log_trade(self, decision, successful=None):
        trade_logs.append({
            'strategy': self.__class__.__name__,
            'time': self.data.datetime.datetime(0),
            'decision': decision,
            'portfolio_value': self.broker.getvalue(),
            'successful': successful
        })

    def next(self):
        position_size = 10

        if self.order:
            return

        if not self.position:
            if self.crossover > 0:  # MACD line crosses above signal line
                self.order = self.buy(size=position_size)
                self.log_trade('buy')
        elif self.crossover < 0:  # MACD line crosses below signal line
            self.order = self.sell(size=position_size)
            self.log_trade('sell')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                profit = order.executed.price - self.buyprice
                successful = profit > 0
                self.log_trade('sell', successful)
        self.order = None

    def stop(self):
        successful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'])
        unsuccessful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'] is False)
        summary_logs.append({
            'symbol': self.data._name,
            'strategy': self.__class__.__name__,
            'starting_value': self.broker.startingcash,
            'ending_value': self.broker.getvalue(),
            'successful_trades': successful_trades,
            'unsuccessful_trades': unsuccessful_trades
        })
        print(f'MACDCrossover Strategy: Successful Trades: {successful_trades}, Unsuccessful Trades: {unsuccessful_trades}')
class OptimizedMACDCrossover(bt.Strategy):
    params = (
        ('macd1', 8),    # Fast EMA period
        ('macd2', 34),    # Slow EMA period
        ('macdsig', 6),   # Signal line period
        ('atr_period', 14),  # ATR for trailing stop
        ('risk_per_trade', 0.02),  # 2% risk per trade
        ('cooldown_period', 30),  # Cooldown period in bars
    )

    def __init__(self):
        # MACD Indicator
        self.macd = bt.indicators.MACD(self.data.close,
                                        period_me1=self.p.macd1,
                                        period_me2=self.p.macd2,
                                        period_signal=self.p.macdsig)
        # ATR Indicator for dynamic stop-loss
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.last_trade_time = None  # To prevent overtrading

    def log_trade(self, decision, successful=None):
        trade_logs.append({
            'strategy': self.__class__.__name__,
            'symbol': self.data._name,
            'time': self.data.datetime.datetime(0),
            'decision': decision,
            'portfolio_value': self.broker.getvalue(),
            'successful': successful
        })

    def calculate_position_size(self):
        cash = self.broker.get_cash()
        risk_amount = cash * self.p.risk_per_trade
        stop_loss_distance = self.atr[0]  # ATR-based stop-loss distance
        position_size = risk_amount / stop_loss_distance
        return int(position_size) if position_size > 0 else 1

    def next(self):
        if self.order:
            return  # Wait for existing orders to complete

        # Enforce Cooldown to avoid overtrading
        if self.last_trade_time and (len(self) - self.last_trade_time) < self.p.cooldown_period:
            return

        # Volume Filter: Enter trades only if current volume > 20-period MA
        vol_ma = np.mean(self.data.volume.get(size=20))
        if self.data.volume[0] <= vol_ma:
            return  # Skip trade if volume is low

        position_size = self.calculate_position_size()

        # MACD Crossover Entry Conditions
        if not self.position:
            if self.macd.macd[0] > self.macd.signal[0]:  # Bullish Crossover
                self.order = self.buy(size=position_size)
                self.buyprice = self.data.close[0]
                self.log_trade('BUY')
                self.last_trade_time = len(self)
            elif self.macd.macd[0] < self.macd.signal[0]:  # Bearish Crossover
                self.order = self.sell(size=position_size)
                self.buyprice = self.data.close[0]
                self.log_trade('SELL')
                self.last_trade_time = len(self)
        else:
            # Implement ATR-based Trailing Stop-Loss and Take-Profit
            if self.position.size > 0:
                stop_loss_price = self.buyprice - self.atr[0]
                take_profit_price = self.buyprice + 2 * self.atr[0]
                if self.data.close[0] <= stop_loss_price:
                    self.close()
                    self.log_trade('STOP_LOSS')
                elif self.data.close[0] >= take_profit_price:
                    self.close()
                    self.log_trade('TAKE_PROFIT')
            elif self.position.size < 0:
                stop_loss_price = self.buyprice + self.atr[0]
                take_profit_price = self.buyprice - 2 * self.atr[0]
                if self.data.close[0] >= stop_loss_price:
                    self.close()
                    self.log_trade('STOP_LOSS')
                elif self.data.close[0] <= take_profit_price:
                    self.close()
                    self.log_trade('TAKE_PROFIT')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                profit = self.buyprice - order.executed.price
                successful = profit > 0
                self.log_trade('SELL', successful)
        self.order = None
    def stop(self):
        successful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'])
        unsuccessful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'] is False)
        summary_logs.append({
            'symbol': self.data._name,
            'strategy': self.__class__.__name__,
            'starting_value': self.broker.startingcash,
            'ending_value': self.broker.getvalue(),
            'successful_trades': successful_trades,
            'unsuccessful_trades': unsuccessful_trades
        })
        print(f'OPTMACDCrossover Strategy: Successful Trades: {successful_trades}, Unsuccessful Trades: {unsuccessful_trades}')
class ContrarianScalping(bt.Strategy):
    params = (('rsi_period', 14), ('overbought', 70), ('oversold', 30), ('stake', 10),)

    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def log_trade(self, decision, successful=None):
        trade_logs.append({
            'strategy': self.__class__.__name__,
            'time': self.data.datetime.datetime(0),
            'decision': decision,
            'portfolio_value': self.broker.getvalue(),
            'successful': successful
        })

    def next(self):
        if self.order:
            return

        position_size = self.p.stake

        if not self.position:
            if self.rsi > self.p.overbought:
                self.order = self.sell(size=position_size)
                self.log_trade('SELL')
            elif self.rsi < self.p.oversold:
                self.order = self.buy(size=position_size)
                self.log_trade('BUY')
        else:
            # Close position when RSI normalizes
            if (self.position.size > 0 and self.rsi > 50) or (self.position.size < 0 and self.rsi < 50):
                self.close()
                self.log_trade('CLOSE')

    def notify_order(self, order):
          if order.status in [order.Completed]:
              if order.isbuy():
                  self.buyprice = order.executed.price  # ✅ Correctly set the buy price
                  self.buycomm = order.executed.comm
                  self.log_trade('BUY')  # Log the BUY action
              elif order.issell():
                  # ✅ Check if buyprice is set before calculating profit
                  if self.buyprice is not None:
                      profit = order.executed.price - self.buyprice
                      successful = profit > 0
                      self.log_trade('SELL', successful)
                  else:
                      # Log that the sell was executed without a prior buy
                      self.log_trade('SELL', successful=False)
          self.order = None


    def stop(self):
        successful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'])
        unsuccessful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'] is False)
        summary_logs.append({
            'symbol': self.data._name,
            'strategy': self.__class__.__name__,
            'starting_value': self.broker.startingcash,
            'ending_value': self.broker.getvalue(),
            'successful_trades': successful_trades,
            'unsuccessful_trades': unsuccessful_trades
        })
        print(f'ContrarianScalping Strategy: Successful Trades: {successful_trades}, Unsuccessful Trades: {unsuccessful_trades}')

class OptimizedContrarianScalping(bt.Strategy):
    params = (('rsi_period', 14), ('overbought', 70), ('oversold', 30), ('stake', 10), ('cooldown_period', 12))

    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        self.last_trade_time = None
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.atr = bt.indicators.ATR(period=14)  # For dynamic stop-loss and take-profit

    def log_trade(self, decision, successful=None):
        trade_logs.append({
            'strategy': self.__class__.__name__,
            'symbol': self.data._name,
            'time': self.data.datetime.datetime(0),
            'decision': decision,
            'portfolio_value': self.broker.getvalue(),
            'successful': successful
        })

    def next(self):
        if self.order:
            return

        # Trade Frequency Control: Cooldown period
        if self.last_trade_time and (len(self) - self.last_trade_time) < self.p.cooldown_period:
            return

        position_size = self.p.stake

        if not self.position:
            # Volume Filter: Only trade if volume exceeds 20-period MA
            vol_ma = np.mean(self.data.volume.get(size=20))
            if self.data.volume[0] > vol_ma:
                if self.rsi[0] > self.p.overbought:
                    self.order = self.sell(size=position_size, exectype=bt.Order.Market,
                                           price=self.data.close[0] * 0.99)  # 1% stop-loss
                    self.log_trade('SELL')
                    self.last_trade_time = len(self)
                elif self.rsi[0] < self.p.oversold:
                    self.order = self.buy(size=position_size, exectype=bt.Order.Market,
                                          price=self.data.close[0] * 1.01)  # 1% stop-loss
                    self.log_trade('BUY')
                    self.last_trade_time = len(self)
        else:
            # ATR-based trailing stop-loss
            stop_loss_price = self.data.close[0] - 1.5 * self.atr[0]
            take_profit_price = self.data.close[0] + 2 * self.atr[0]
            if self.data.close[0] <= stop_loss_price:
                self.close()
                self.log_trade('STOP_LOSS')
            elif self.data.close[0] >= take_profit_price:
                self.close()
                self.log_trade('TAKE_PROFIT')
    def stop(self):
        successful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'])
        unsuccessful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'] is False)
        summary_logs.append({
            'symbol': self.data._name,
            'strategy': self.__class__.__name__,
            'starting_value': self.broker.startingcash,
            'ending_value': self.broker.getvalue(),
            'successful_trades': successful_trades,
            'unsuccessful_trades': unsuccessful_trades
        })
        print(f'OptzContrarianScalping Strategy: Successful Trades: {successful_trades}, Unsuccessful Trades: {unsuccessful_trades}')

class VolatilityBreakout(bt.Strategy):
    params = (('period', 20), ('devfactor', 2), ('stake', 10),)

    def __init__(self):
        self.boll = bt.indicators.BollingerBands(period=self.p.period, devfactor=self.p.devfactor)
        self.order = None
        self.buyprice = None
        self.buycomm = None

    def log_trade(self, decision, successful=None):
        trade_logs.append({
            'strategy': self.__class__.__name__,
            'time': self.data.datetime.datetime(0),
            'decision': decision,
            'portfolio_value': self.broker.getvalue(),
            'successful': successful
        })

    def next(self):
        if self.order:
            return

        position_size = self.p.stake

        # Volume confirmation with last 5 bars
        recent_volumes = list(self.data.volume.get(size=5))

        if not self.position:
            # Breakout above upper band with volume spike
            if self.data.close[0] > self.boll.lines.top[0] and self.data.volume[0] > max(recent_volumes):
                self.order = self.buy(size=position_size)
                self.log_trade('BUY')
            # Breakout below lower band with volume spike
            elif self.data.close[0] < self.boll.lines.bot[0] and self.data.volume[0] > max(recent_volumes):
                self.order = self.sell(size=position_size)
                self.log_trade('SELL')
        else:
            # Close position if price moves back inside the bands
            if (self.position.size > 0 and self.data.close[0] < self.boll.lines.mid[0]) or \
               (self.position.size < 0 and self.data.close[0] > self.boll.lines.mid[0]):
                self.close()
                self.log_trade('CLOSE')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                profit = order.executed.price - self.buyprice
                successful = profit > 0
                self.log_trade('SELL', successful)
        self.order = None

    def stop(self):
        successful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'])
        unsuccessful_trades = sum(1 for log in trade_logs if log['strategy'] == self.__class__.__name__ and log['successful'] is False)
        summary_logs.append({
            'symbol': self.data._name,
            'strategy': self.__class__.__name__,
            'starting_value': self.broker.startingcash,
            'ending_value': self.broker.getvalue(),
            'successful_trades': successful_trades,
            'unsuccessful_trades': unsuccessful_trades
        })
        print(f'VolatilityBreakout Strategy: Successful Trades: {successful_trades}, Unsuccessful Trades: {unsuccessful_trades} on symbol {self.data._name}')



def save_to_csv(df, filename):
    # Check if the file exists
    file_exists = os.path.isfile(filename)
    
    # Write to the file in append mode if it exists, else create a new file
    df.to_csv(filename, mode='a', index=False, header=not file_exists)
def extract_trade_data(results):
    strategy = results[0]
    trade_analyzer = strategy.analyzers.trade_analyzer.get_analysis()

    trades = []
    for key, value in trade_analyzer.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                trades.append({
                    'Metric': f'{key}_{nested_key}',
                    'Value': nested_value
                })
        else:
            trades.append({
                'Metric': key,
                'Value': value
            })

    return pd.DataFrame(trades)



# Function to run backtests for each strategy
def run_backtest(strategy, symbols, start_date, end_date, interval='5m'):
  for symbol in symbols:
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy)

    # Fetch historical data
    data = fetch_historical_data(symbol, start_date, end_date, interval)
    data_feed = bt.feeds.PandasData(dataname=data, name=symbol)
    cerebro.adddata(data_feed)

    # Set initial cash and commission
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.005)

    print(f"\nRunning {strategy.__name__} for {symbol} from {start_date} to {end_date}")
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Ending Portfolio Value: {cerebro.broker.getvalue():.2f}")



# Main execution block
if __name__ == '__main__':
  # Define symbols and run backtests
  symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'AMD', 'INTC']
  end_date = datetime.today()
  start_date = end_date - timedelta(days=58)  # Run the backtest for 20 days

  # List of strategies to test
  strategies = [BollingerROC,  MovingAverageCrossover, MACDCrossover,VolatilityBreakout,ContrarianScalping,OptimizedContrarianScalping,OptimizedMACDCrossover]

  # Run backtest for each strategy
  for strategy in strategies:
      run_backtest(strategy, symbols, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
      trade_df = pd.DataFrame(trade_logs)
      summary_df = pd.DataFrame(summary_logs)
      save_to_csv(trade_df, 'backtest_results.csv')
      save_to_csv(summary_df, 'backtest_summary.csv')
    
