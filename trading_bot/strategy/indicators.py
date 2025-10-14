import pandas as pd
import numpy as np

class TechnicalIndicators:
    @staticmethod
    def calculate_ema(data, period):
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def add_all_indicators(df):
        if df.empty or len(df) < 50:
            return df
        df = df.copy()
        df['ema_9'] = TechnicalIndicators.calculate_ema(df['close'], 9)
        df['ema_21'] = TechnicalIndicators.calculate_ema(df['close'], 21)
        df['ema_50'] = TechnicalIndicators.calculate_ema(df['close'], 50)
        df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'], 14)
        df['atr'] = TechnicalIndicators.calculate_atr(df['high'], df['low'], df['close'])
        return df