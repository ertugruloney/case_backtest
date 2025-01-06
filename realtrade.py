

import pandas as pd
import numpy as np
import pandas_ta as ta
from grph import TradingChart  # Assuming this is your custom plotting module
import datetime
import ccxt
import pandas as pd
from datetime import datetime
import telegram
import time
import traceback
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
        self.position=0
        self.binance = ccxt.binance({
            'apiKey':'o5IusN1vTt44WWiTCOdKh2UKcL0ilhhNUKKPSVeu6RZ0qfHxu2NgkAIv9g0ZH4ST' ,
            'secret':'oAfsnjEQjmjxCgBOyMAbq9xW73SGTXybOUyOvv9wKhpMPdeAdfGg6yzQnAlCqvRr' ,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}  # Futures işlemleri için
    
        })
    

        self.leverage=2
        self.symbol= 'BTC/USDT'
        self.size=0.005
        self.telegram_bot = telegram.Bot(token='8127804933:AAECpSP7l2wHGeS1Jlcq5QRaDctxg0E0y8k')  # Telegram bot token
        self.chat_id = '7894595813'  # Telegram chat ID
    def keltner_channel(self, df):
        """Calculates the Keltner Channel."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        middle_line = typical_price.rolling(window=self.KC_PERIOD).mean()
        tr = df.ta.true_range()
        atr = tr.rolling(window=self.KC_PERIOD).mean()
        upper_band = middle_line + self.KC_MULTIPLIER * atr
        lower_band = middle_line - self.KC_MULTIPLIER * atr
        return upper_band, middle_line, lower_band
    def set_leverage(self,):
        markets =self.binance.load_markets()
        market = markets[self.symbol]
        self.binance.set_leverage(self.leverage, self.symbol) 
    def send_telegram_message(self, message):
        """Sends a message to Telegram."""
        try:
            self.telegram_bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            print(f"Telegram notification failed: {str(e)}")

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
        try:
                self.set_leverage()
                entry_price = 0
                stop_loss = 0
                take_profit = 0
                entry_bar = 0
                trades = []
        
                i=len(df)-1 
             
                
                if  self.position == 1:
                    exit_price = df['Close'][i]
                    trades[-1]['Exit Time'] = df['timestamp'][i]
                    trades[-1]['Exit Price'] = exit_price
                    trades[-1]['Profit'] = (exit_price - entry_price) / entry_price * 100
                    self.binance.create_market_sell_order(self.symbol,self.size)
                    self.position = 0
                    self.send_telegram_message(f"long çikildi: {self.symbol}")
                elif self.position == -1:
                    exit_price = df['Close'][i]
                    trades[-1]['Exit Time'] = df['timestamp'][i]
                    trades[-1]['Exit Price'] = exit_price
                    trades[-1]['Profit'] = (entry_price - exit_price) / entry_price * 100
                    self.binance.create_market_buy_order(self.symbol,self.size)
                    self.position = 0
                    self.send_telegram_message(f"short çikildi: {self.symbol}")
        
                # Trading logic
                if self.position == 0:
                    if df['LongEntrySignal'][i]:
                        self.position = 1
                        self.binance.create_market_buy_order(self.symbol,self.size)
                      
                        stop_loss = entry_price * (1 - self.SL_PERCENT / 100)
                        take_profit = entry_price * (1 + self.TP_PERCENT / 100)
                        self.send_telegram_message(f"long girildi: {self.symbol}")
                        trades.append({'Entry Time': df['timestamp'][i], 'Entry Price': entry_price, 'Direction': 'Long', 'Stop Loss': stop_loss, 'Take Profit': take_profit})
                    elif df['ShortEntrySignal'][i]:
                        self.position = -1
                        self.binance.create_market_sell_order(self.symbol,self.size)
                        entry_price = df['Open'][i]
                        stop_loss = entry_price * (1 + self.SL_PERCENT / 100)
                        take_profit = entry_price * (1 - self.TP_PERCENT / 100)
                        entry_bar = i
                        self.send_telegram_message(f"short girildi: {self.symbol}")
                        trades.append({'Entry Time': df['timestamp'][i], 'Entry Price': entry_price, 'Direction': 'Short', 'Stop Loss': stop_loss, 'Take Profit': take_profit})
                elif self.position == 1:
                    if df['LongExitSignal'][i] or df['Low'][i] <= stop_loss or df['High'][i] >= take_profit or (i - entry_bar) >= self.EXIT_AFTER_BARS:
                        exit_price = df['Close'][i] if not (df['Low'][i] <= stop_loss or df['High'][i] >= take_profit) else (stop_loss if df['Low'][i] <= stop_loss else take_profit)
                        trades[-1]['Exit Time'] = df['timestamp'][i]
                        trades[-1]['Exit Price'] = exit_price
                        trades[-1]['Profit'] = (exit_price - entry_price) / entry_price * 100
                        self.binance.create_market_sell_order(self.symbol,self.size)
                        self.position = 0
                        self.send_telegram_message(f"long çikildi: {self.symbol}")
                elif self.position == -1:
                    if df['ShortExitSignal'][i] or df['High'][i] >= stop_loss or df['Low'][i] <= take_profit or (i - entry_bar) >= self.EXIT_AFTER_BARS:
                        exit_price = df['Close'][i] if not (df['High'][i] >= stop_loss or df['Low'][i] <= take_profit) else (stop_loss if df['High'][i] >= stop_loss else take_profit)
                        trades[-1]['Exit Time'] = df['timestamp'][i]
                        trades[-1]['Exit Price'] = exit_price
                        trades[-1]['Profit'] = (entry_price - exit_price) / entry_price * 100
                        self.binance.create_market_buy_order(self.symbol,self.size)
                        self.position = 0
                        self.send_telegram_message(f"short çikildi: {self.symbol}")
        except Exception as e:
            error_message = f"Error in backtest_strategy: {str(e)}\n{traceback.format_exc()}"
            self.send_telegram_message(error_message)
            print( error_message)
    

    def run_strategy(self,):
        """Runs the complete strategy."""
        
        while True:
            try:
                today = datetime.today()
          

                print(f'bot çalışıyor {today}')
            
                timeframe = '1h'
        
                # Son 50 mum verisini çek
                limit = 50
                ohlcv = self.binance.fetch_ohlcv(self.symbol, timeframe, limit=limit)
        
                # Veriyi DataFrame'e çevir
                columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
                df = pd.DataFrame(ohlcv, columns=columns)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = self.calculate_indicators(df.copy())
                df = self.generate_signals(df)
                self.backtest_strategy(df)
                time.sleep(300)
                print('kontrol bitti')
            except Exception as e:
                error_message = f"Error in run strategy: {str(e)}\n{traceback.format_exc()}"
                self.send_telegram_message(error_message)
                print( error_message)


# Example usage
if __name__ == "__main__":
    
    hiperparameters=pd.read_csv('Hiperparameters.csv').values
    
    strategy = KeltnerChannelStrategy(KC_PERIOD=int(hiperparameters[0][1]),
                                      ATR_PERIOD=int(hiperparameters[1][1]),
                                      LWMA_PERIOD=int(hiperparameters[2][1]),
                                      TP_PERCENT=hiperparameters[3][1],
                                      SL_PERCENT=hiperparameters[4][1],
                                      EXIT_AFTER_BARS=int(hiperparameters[5][1]))
    

    strategy.run_strategy()
    


