import ccxt
import pandas as pd
from datetime import datetime
import time
import time
from datetime import datetime, timedelta
# Binance borsasına bağlan
binance = ccxt.binance({'options': {
        'defaultType': 'future'  # Futures verisi için
    }})

def fetch_ohlcv_with_pagination(exchange, symbol, start_time, end_time, timeframe='1m', limit=1000):
    """Veri çekme süresini bölerek zaman dilimleri arasında ilerleyerek veri çeker."""
    all_candles = []
    while start_time < end_time:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=start_time, limit=limit)
            if not candles:
                break
            all_candles.extend(candles)
            # Son mumun timestamp'ını alarak bir sonraki istek için yeni start_time olarak kullanıyoruz
            start_time = candles[-1][0] + 1  # Yeni başlangıç zamanı, son mumun timestamp'ı + 1ms
            time.sleep(1)  # Rate limiting'e karşı kısa bir bekleme süresi
        except Exception as e:
            print(f"Error fetching {symbol} from {exchange.id}: {e}")
            break
    return all_candles

timeframe = '1h'
days = 30
# Kullanılacak borsalar
  
# Zaman aralıkları
end_time = int(datetime.utcnow().timestamp() * 1000)  # Şu anki zaman
start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)  # 6 gün önce


markets = binance.load_markets()

# USDT bazlı coinleri filtrele
usdt_pairs = [market for market in markets if market.endswith('/USDT')]


for symbol in usdt_pairs:
        exchange_name="binance"
        exchange = getattr(ccxt, exchange_name)()
        exchange.load_markets()
        a=symbol.split("/")
    

        print(f"{exchange_name} - {symbol} verileri çekiliyor...")
        candles = fetch_ohlcv_with_pagination(exchange, symbol, start_time, end_time, timeframe)
        if candles:
            # Veriyi pandas DataFrame'e çevirme ve CSV'ye kaydetme
            df = pd.DataFrame(candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            filename = f"{a[0]}.csv"
            filename = "coins/" + filename
            df.to_csv(filename, index=False)
            print(f"{symbol} için veri kaydedildi: {filename}")
        else:
            print(f"{symbol} için veri bulunamadı.")