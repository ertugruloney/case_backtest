#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 03:05:40 2024

@author: ertugruloney
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from grph import TradingChart  # Assuming this is your custom plotting module
import datetime

class KeltnerChannelStrategy:
    """
    A class to represent a Keltner Channel trading strategy with 
    backtesting and plotting capabilities.
    """

    def __init__(self, KC_PERIOD, ATR_PERIOD, LWMA_PERIOD, 
                 EXIT_AFTER_BARS, TP_PERCENT, SL_PERCENT, 
                 KC_MULTIPLIER=2.5):
        """
        Initializes the KeltnerChannelStrategy with default parameters.
        """
        self.KC_PERIOD = KC_PERIOD
        self.ATR_PERIOD = ATR_PERIOD
        self.LWMA_PERIOD = LWMA_PERIOD
        self.EXIT_AFTER_BARS = EXIT_AFTER_BARS
        self.TP_PERCENT = TP_PERCENT
        self.SL_PERCENT = SL_PERCENT
        self.KC_MULTIPLIER = KC_MULTIPLIER

    def keltner_channel(self, df):
        """Calculates the Keltner Channel."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        middle_line = typical_price.rolling(window=self.KC_PERIOD).mean()
        tr = df.ta.true_range()
        atr = tr.rolling(window=self.KC_PERIOD).mean()
        upper_band = middle_line + self.KC_MULTIPLIER * atr
        lower_band = middle_line - self.KC_MULTIPLIER * atr
        return upper_band, middle_line, lower_band

    def calculate_indicators(self, df):
        """Calculates technical indicators."""
        df['KC_Upper'], df['KC_Middle'], df['KC_Lower'] = self.keltner_channel(df)
        df.ta = ta.Strategy(df)
        df['TrueRange'] = ta.true_range(df['High'], df['Low'], df['Close'])
        df['ATR'] = df['TrueRange'].rolling(window=self.ATR_PERIOD).mean()
        df['LWMA'] = df['Close'].ewm(span=self.LWMA_PERIOD, adjust=False).mean()
        return df

    def generate_signals(self, df):
        """Generates trading signals."""
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

    def backtest_strategy(self, df):
        """Backtests the trading strategy."""
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
                    stop_loss = entry_price * (1 - self.SL_PERCENT / 100)
                    take_profit = entry_price * (1 + self.TP_PERCENT / 100)
                    entry_bar = i
                    trades.append({'Entry Time': df['timestamp'][i], 'Entry Price': entry_price, 'Direction': 'Long', 'Stop Loss': stop_loss, 'Take Profit': take_profit})
                elif df['ShortEntrySignal'][i]:
                    position = -1
                    entry_price = df['Open'][i]
                    stop_loss = entry_price * (1 + self.SL_PERCENT / 100)
                    take_profit = entry_price * (1 - self.TP_PERCENT / 100)
                    entry_bar = i
                    trades.append({'Entry Time': df['timestamp'][i], 'Entry Price': entry_price, 'Direction': 'Short', 'Stop Loss': stop_loss, 'Take Profit': take_profit})
            elif position == 1:
                if df['LongExitSignal'][i] or df['Low'][i] <= stop_loss or df['High'][i] >= take_profit or (i - entry_bar) >= self.EXIT_AFTER_BARS:
                    exit_price = df['Close'][i] if not (df['Low'][i] <= stop_loss or df['High'][i] >= take_profit) else (stop_loss if df['Low'][i] <= stop_loss else take_profit)
                    trades[-1]['Exit Time'] = df['timestamp'][i]
                    trades[-1]['Exit Price'] = exit_price
                    trades[-1]['Profit'] = (exit_price - entry_price) / entry_price * 100
                    position = 0
            elif position == -1:
                if df['ShortExitSignal'][i] or df['High'][i] >= stop_loss or df['Low'][i] <= take_profit or (i - entry_bar) >= self.EXIT_AFTER_BARS:
                    exit_price = df['Close'][i] if not (df['High'][i] >= stop_loss or df['Low'][i] <= take_profit) else (stop_loss if df['High'][i] >= stop_loss else take_profit)
                    trades[-1]['Exit Time'] = df['timestamp'][i]
                    trades[-1]['Exit Price'] = exit_price
                    trades[-1]['Profit'] = (entry_price - exit_price) / entry_price * 100
                    position = 0
        return pd.DataFrame(trades)

    def run_strategy(self, df):
        """Runs the complete strategy."""
        df = self.calculate_indicators(df.copy())
        df = self.generate_signals(df)
        trades_df = self.backtest_strategy(df)
        return trades_df.dropna()

    def plot_results(self, price_data, trade_data):
        """Plots the trading results."""
        chart = TradingChart(price_data, trade_data)
        chart.plot(point_size=150)

# Example usage
if __name__ == "__main__":
    
    hiperparameters=pd.read_csv('hiperparameters.csv').values
    
    strategy = KeltnerChannelStrategy(KC_PERIOD=int(hiperparameters[0][1]),
                                      ATR_PERIOD=int(hiperparameters[1][1]),
                                      LWMA_PERIOD=int(hiperparameters[2][1]),
                                      TP_PERCENT=hiperparameters[3][1],
                                      SL_PERCENT=hiperparameters[4][1],
                                      EXIT_AFTER_BARS=int(hiperparameters[5][1]))
    
    """
    strategy = KeltnerChannelStrategy(KC_PERIOD=20,
                                      ATR_PERIOD=40,
                                      LWMA_PERIOD=20,
                                      TP_PERCENT=3.9,
                                      SL_PERCENT=8.2,
                                      EXIT_AFTER_BARS=253)
"""
    # Load data
    df = pd.read_csv("btc_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Run strategy
    trades_df = strategy.run_strategy(df)

    # Print and save results
    print("Trades:")
    print(trades_df.to_markdown(index=False, numalign="left", stralign="left"))
    trades_df.to_csv('trade.csv')

    # Calculate and print the total profit
    total_profit = trades_df['Profit'].sum()
    print(f"\nTotal Profit: {total_profit:.2f}%")

    # Plot results
    strategy.plot_results(df, trades_df)