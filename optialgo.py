import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.optimize import differential_evolution
import datetime

class KeltnerChannelStrategy:
    """
    A class to represent a Keltner Channel trading strategy with 
    parameter optimization and backtesting capabilities.
    """

    def __init__(self, KC_MULTIPLIER=2.5):
        """
        Initializes the KeltnerChannelStrategy with a default 
        KC_MULTIPLIER.
        """
        self.KC_MULTIPLIER = KC_MULTIPLIER

    def keltner_channel(self, df, period, multiplier):
        """
        Calculates the Keltner Channel.

        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data.
            period (int): Period for the Keltner Channel calculation.
            multiplier (float): Multiplier for the Average True Range.

        Returns:
            tuple: Upper, Middle, and Lower bands of the Keltner Channel.
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        middle_line = typical_price.rolling(window=period).mean()
        tr = df.ta.true_range()
        atr = tr.rolling(window=period).mean()
        upper_band = middle_line + multiplier * atr
        lower_band = middle_line - multiplier * atr
        return upper_band, middle_line, lower_band

    def calculate_indicators(self, df, KC_PERIOD, ATR_PERIOD, LWMA_PERIOD):
        """
        Calculates technical indicators required for the trading strategy.

        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data.

        Returns:
            pd.DataFrame: DataFrame with added indicator columns.
        """
        df['KC_Upper'], df['KC_Middle'], df['KC_Lower'] = self.keltner_channel(df, KC_PERIOD, self.KC_MULTIPLIER)
        df.ta = ta.Strategy(
            name="MyStrategy",
            ta=[
                {"kind": "true_range"},
            ]
        )
        df['TrueRange'] = ta.true_range(df['High'], df['Low'], df['Close'])
        df['ATR'] = df['TrueRange'].rolling(window=ATR_PERIOD).mean()
        df['LWMA'] = df['Close'].ewm(span=LWMA_PERIOD, adjust=False).mean()
        return df

    def generate_signals(self, df):
        """
        Generates trading signals based on the strategy logic.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV and indicator data.

        Returns:
            pd.DataFrame: DataFrame with added signal columns.
        """
        df['LongEntrySignal'] = (df['Open'] > df['KC_Lower'].shift(3)) & (df['Open'].shift(3) <= df['KC_Lower'].shift(6))
        df['ShortEntrySignal'] = (df['Open'] < df['KC_Upper'].shift(3)) & (df['Open'].shift(3) >= df['KC_Upper'].shift(6))
        df['LongExitSignal'] = (
            (df['TrueRange'].shift(2) > df['TrueRange'].shift(3))
            & (df['TrueRange'].shift(3) > df['TrueRange'].shift(4))
            & (df['TrueRange'].shift(4) > df['TrueRange'].shift(5)) 
            & (df['ATR'].shift(1) < df['ATR'].shift(2))  
            & (df['Close'].shift(3) < df['LWMA'].shift(3))  
        )
        df['ShortExitSignal'] = (
            (df['TrueRange'].shift(2) < df['TrueRange'].shift(3))
            & (df['TrueRange'].shift(3) < df['TrueRange'].shift(4))
            & (df['TrueRange'].shift(4) < df['TrueRange'].shift(5))  
            & (df['ATR'].shift(1) < df['ATR'].shift(2)) 
            & (df['Close'].shift(3) > df['LWMA'].shift(3))  
        )
        return df

    def backtest_strategy(self, df, TP_PERCENT, SL_PERCENT, EXIT_AFTER_BARS):
        """
        Backtests the trading strategy on the provided data.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV, indicator, and signal data.

        Returns:
            pd.DataFrame: DataFrame with trade details.
        """
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        entry_bar = 0
        trades = []

        for i in range(6, len(df)):
            current_time = df['timestamp'][i].time()
            current_day = df['timestamp'][i].weekday()

            # Apply trading time restrictions
            if current_day == 5 and current_time >= datetime.time(23, 0):  # Friday after 23:00
                continue
            if current_day == 6:  # Saturday
                continue
            if current_day == 0 and current_time < datetime.time(0, 0):  # Sunday before 00:00
                continue

            # Friday exit
            if current_day == 4 and current_time >= datetime.time(20, 40):  # Friday after 20:40
                if position == 1:
                    exit_price = df['Close'][i]
                    trades[-1]['Exit Time'] = df['timestamp'][i]
                    trades[-1]['Exit Price'] = exit_price
                    trades[-1]['Profit'] = (exit_price - entry_price) / entry_price * 100
                    position = 0
                elif position == -1:
                    exit_price = df['Close'][i]
                    trades[-1]['Exit Time'] = df['timestamp'][i]
                    trades[-1]['Exit Price'] = exit_price
                    trades[-1]['Profit'] = (entry_price - exit_price) / entry_price * 100
                    position = 0
                continue

            # Trading logic
            if position == 0:
                if df['LongEntrySignal'][i]:
                    position = 1
                    entry_price = df['Open'][i]
                    stop_loss = entry_price * (1 - SL_PERCENT / 100)
                    take_profit = entry_price * (1 + TP_PERCENT / 100)
                    entry_bar = i
                    trades.append({'Entry Time': df['timestamp'][i], 'Entry Price': entry_price, 'Direction': 'Long', 'Stop Loss': stop_loss, 'Take Profit': take_profit})
                elif df['ShortEntrySignal'][i]:
                    position = -1
                    entry_price = df['Open'][i]
                    stop_loss = entry_price * (1 + SL_PERCENT / 100)
                    take_profit = entry_price * (1 - TP_PERCENT / 100)
                    entry_bar = i
                    trades.append({'Entry Time': df['timestamp'][i], 'Entry Price': entry_price, 'Direction': 'Short', 'Stop Loss': stop_loss, 'Take Profit': take_profit})
            elif position == 1:
                if df['LongExitSignal'][i] or df['Low'][i] <= stop_loss or df['High'][i] >= take_profit or (i - entry_bar) >= EXIT_AFTER_BARS:
                    exit_price = df['Close'][i] if not (df['Low'][i] <= stop_loss or df['High'][i] >= take_profit) else (stop_loss if df['Low'][i] <= stop_loss else take_profit)
                    trades[-1]['Exit Time'] = df['timestamp'][i]
                    trades[-1]['Exit Price'] = exit_price
                    trades[-1]['Profit'] = (exit_price - entry_price) / entry_price * 100
                    position = 0
            elif position == -1:
                if df['ShortExitSignal'][i] or df['High'][i] >= stop_loss or df['Low'][i] <= take_profit or (i - entry_bar) >= EXIT_AFTER_BARS:
                    exit_price = df['Close'][i] if not (df['High'][i] >= stop_loss or df['Low'][i] <= take_profit) else (stop_loss if df['High'][i] >= stop_loss else take_profit)
                    trades[-1]['Exit Time'] = df['timestamp'][i]
                    trades[-1]['Exit Price'] = exit_price
                    trades[-1]['Profit'] = (entry_price - exit_price) / entry_price * 100
                    position = 0
        return pd.DataFrame(trades)

    def objective_function(self, params, df):
        """
        Objective function to be minimized.

        Args:
            params (list): List of parameters to be optimized.
                params[0]: KC_PERIOD
                params[1]: ATR_PERIOD
                params[2]: LWMA_PERIOD
                params[3]: TP_PERCENT
                params[4]: SL_PERCENT
                params[5]: EXIT_AFTER_BARS
            df (pd.DataFrame): DataFrame with OHLCV data.

        Returns:
            float: Negative of the total profit (to be minimized).
        """
        KC_PERIOD = int(params[0])
        ATR_PERIOD = int(params[1])
        LWMA_PERIOD = int(params[2])
        TP_PERCENT = params[3]
        SL_PERCENT = params[4]
        EXIT_AFTER_BARS = int(params[5])

        df = self.calculate_indicators(df.copy(), KC_PERIOD, ATR_PERIOD, LWMA_PERIOD)
        df = self.generate_signals(df)
        trades_df = self.backtest_strategy(df, TP_PERCENT, SL_PERCENT, EXIT_AFTER_BARS)
        trades_df = trades_df.dropna()
        total_profit = trades_df['Profit'].sum()
        return -total_profit

    def optimize_parameters(self, df, bounds):
        """
        Performs differential evolution optimization to find the best 
        strategy parameters.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
            bounds (list): List of tuples defining the bounds for each parameter.

        Returns:
            scipy.optimize.OptimizeResult: Result of the optimization.
        """
        result = differential_evolution(self.objective_function, bounds, args=(df,), maxiter=50, popsize=15)
        return result

    def run_strategy(self, df, params):
        """
        Runs the strategy with the given parameters and returns the trades.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
            params (list): List of optimized parameters.

        Returns:
            pd.DataFrame: DataFrame with trade details.
        """
        df = self.calculate_indicators(df.copy(), int(params[0]), int(params[1]), int(params[2]))
        df = self.generate_signals(df)
        trades_df = self.backtest_strategy(df, params[3], params[4], int(params[5]))
        trades_df = trades_df.dropna()
        return trades_df

# Example usage
if __name__ == "__main__":
    print("Optimization Algorithm Started.")
    strategy = KeltnerChannelStrategy()

    # Load data
    df = pd.read_csv("btc_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Define bounds for optimization
    bounds = [
        (10, 50),  # KC_PERIOD
        (20, 80),  # ATR_PERIOD
        (10, 50),  # LWMA_PERIOD
        (5, 20),  # TP_PERCENT
        (5, 20),  # SL_PERCENT
        (50, 300),  # EXIT_AFTER_BARS
    ]

    # Optimize parameters
    result = strategy.optimize_parameters(df, bounds)
    
#%%
    columns=['KC_PERIOD','ATR_PERIOD','LWMA_PERIOD','TP_PERCENT','SL_PERCENT','EXIT_AFTER_BARS']
    result_df=[]
    for i in range(6):
        if i !=4 and i!=3:
            result_df.append(int(result.x[i]) )
            
        else:
            result_df.append(float(result.x[i]))
    
    result_df=pd.DataFrame(result_df,index=columns, columns=['Values'])
    result_df.to_csv('Hiperparameters.csv')

    # Print optimized parameters
    print("Optimized Parameters:")
    print(f"KC_PERIOD: {int(result.x[0])}")
    print(f"ATR_PERIOD: {int(result.x[1])}")
    print(f"LWMA_PERIOD: {int(result.x[2])}")
    print(f"TP_PERCENT: {result.x[3]:.2f}")
    print(f"SL_PERCENT: {result.x[4]:.2f}")
    print(f"EXIT_AFTER_BARS: {int(result.x[5])}")



  
    print("Optimization Algorithm Finished.")
   