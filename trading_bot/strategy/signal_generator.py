import pandas as pd
from dataclasses import dataclass
from .indicators import TechnicalIndicators
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger('signal_generator')

@dataclass
class TradingSignal:
    symbol: str
    side: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size: float
    timestamp: pd.Timestamp

class SignalGenerator:
    def generate_signal(self, symbol, data, current_price):
        primary_tf = Settings.PRIMARY_TIMEFRAME
        if primary_tf not in data or data[primary_tf].empty:
            return None
        
        df = TechnicalIndicators.add_all_indicators(data[primary_tf])
        
        # Detecção simples de reversão
        score = 0
        latest = df.iloc[-1]
        
        # RSI oversold/overbought
        if latest['rsi'] < 30:
            score += 1
        elif latest['rsi'] > 70:
            score -= 1
        
        # EMA cross
        if latest['ema_9'] > latest['ema_21']:
            score += 0.5
        else:
            score -= 0.5
        
        if abs(score) < 0.8:
            return None
        
        side = 'long' if score > 0 else 'short'
        atr = latest['atr']
        
        if side == 'long':
            stop_loss = current_price - (atr * Settings.SL_ATR_MULTIPLIER)
            tp1 = current_price + (atr * Settings.TP1_ATR_MULTIPLIER)
            tp2 = current_price + (atr * Settings.TP2_ATR_MULTIPLIER)
        else:
            stop_loss = current_price + (atr * Settings.SL_ATR_MULTIPLIER)
            tp1 = current_price - (atr * Settings.TP1_ATR_MULTIPLIER)
            tp2 = current_price - (atr * Settings.TP2_ATR_MULTIPLIER)
        
        signal = TradingSignal(
            symbol=symbol,
            side=side,
            confidence=abs(score),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            position_size=0,
            timestamp=pd.Timestamp.now()
        )
        
        logger.info(f"Signal: {side.upper()} {symbol} @ {current_price}")
        return signal