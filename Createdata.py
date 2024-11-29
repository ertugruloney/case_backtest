import ccxt
import pandas as pd
from datetime import datetime

# Binance borsasına bağlan
binance = ccxt.binance()

# Tarihleri belirle (Kasım ayı için)
start_date = '2024-11-01 00:00:00'
end_date = '2024-11-30 23:59:59'

# Timestamps'e çevir
start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)

# Veri çekme
symbol = 'BTC/USDT'
timeframe = '1h'

ohlcv = []
while start_timestamp < end_timestamp:
    data = binance.fetch_ohlcv(symbol, timeframe, since=start_timestamp, limit=1000)
    if not data:
        break
    ohlcv += data
    start_timestamp = data[-1][0] + 3600000  # 1 saat ekle

# DataFrame'e çevir
columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
df = pd.DataFrame(ohlcv, columns=columns)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Dosyaya kaydet
df.to_csv('btc_data.csv', index=False) 
print("Veriler başarıyla kaydedildi!")