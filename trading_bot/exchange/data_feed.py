import pandas as pd
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger('data_feed')

class DataFeed:
    def __init__(self, exchange):
        self.exchange = exchange
        self.cache = {}
    
    def fetch_ohlcv(self, symbol, timeframe_minutes, limit=500):
        try:
            timeframe_str = self._get_timeframe_string(timeframe_minutes)
            ohlcv = self.exchange.exchange.fetch_ohlcv(symbol, timeframe_str, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe_minutes}m")
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            return pd.DataFrame()
    
    def get_latest_candles(self, symbol, timeframe_minutes, count=100):
        return self.fetch_ohlcv(symbol, timeframe_minutes, limit=count)
    
    def get_multiple_timeframes(self, symbol, timeframes, count=100):
        result = {}
        for tf in timeframes:
            df = self.get_latest_candles(symbol, tf, count)
            if not df.empty:
                result[tf] = df
        return result
    
    @staticmethod
    def _get_timeframe_string(minutes):
        if minutes < 60:
            return f"{minutes}m"
        elif minutes < 1440:
            return f"{minutes // 60}h"
        else:
            return f"{minutes // 1440}d" 