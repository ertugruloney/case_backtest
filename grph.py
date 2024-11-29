import pandas as pd
import matplotlib.pyplot as plt

class TradingChart:
    def __init__(self, price_data, trade_data):
        """
        Initialize the TradingChart class.
        :param price_data: DataFrame with price data (must include 'timestamp' and 'close' columns).
        :param trade_data: DataFrame with trade data (must include 'Entry Time', 'Exit Time', and 'Direction').
        """
        self.price_data = price_data
        self.trade_data = trade_data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data by converting timestamps and setting indices."""
        self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
        self.trade_data['Entry Time'] = pd.to_datetime(self.trade_data['Entry Time'])
        self.trade_data['Exit Time'] = pd.to_datetime(self.trade_data['Exit Time'])
        self.price_data.set_index('timestamp', inplace=True)

    def plot(self, point_size=100):
        """
        Plot the trading chart with buy and sell points.
        :param point_size: Size of the buy/sell markers on the chart.
        """
        plt.figure(figsize=(14, 8))
        plt.plot(self.price_data.index, self.price_data['Close'], label='BTC Close Price', color='blue')

        for _, trade in self.trade_data.iterrows():
            entry_time = trade['Entry Time']
            exit_time = trade['Exit Time']
            direction = trade['Direction']
            entry_price = self.price_data.loc[entry_time, 'Open'] if entry_time in self.price_data.index else None
            exit_price = self.price_data.loc[exit_time, 'Close'] if exit_time in self.price_data.index else None

            if entry_price is not None and exit_price is not None:
                if direction == 'Long':
                    plt.scatter(entry_time, entry_price, color='green', s=point_size, label='Buy' if 'Buy' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.scatter(exit_time, exit_price, color='red', s=point_size, label='Sell' if 'Sell' not in plt.gca().get_legend_handles_labels()[1] else "")
                else:
                    plt.scatter(entry_time, entry_price, color='orange', s=point_size, label='Short Entry' if 'Short Entry' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.scatter(exit_time, exit_price, color='purple', s=point_size, label='Short Exit' if 'Short Exit' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.title('BTC Trading Strategy Simulation')
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.savefig('trading_chart.png')
        plt.show()
