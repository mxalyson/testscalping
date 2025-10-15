"""
Strategy Core - Detecção de Topos e Fundos (VERSÃO CORRIGIDA)
==============================================================

Estratégia de scalping baseada em:
- Detecção de swing highs/lows
- Confluência de indicadores técnicos (EMA, RSI, MACD, ATR)
- Padrões de candle (engulfing, pinbar, doji)
- Confirmação multi-timeframe
- Filtros de volatilidade

✅ CORREÇÕES APLICADAS:
- Sistema de pontos mais generoso
- Pontos parciais adicionados
- Logs de debug melhorados
- Threshold reduzido para 50%

Autor: Trading Bot Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import ta


# ============================================================================
# ENUMS E CONSTANTES
# ============================================================================

class SignalType(Enum):
    """Tipos de sinal de trading"""
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    NEUTRAL = "NEUTRAL"


class CandlePattern(Enum):
    """Padrões de candle identificados"""
    BULLISH_ENGULFING = "BULLISH_ENGULFING"
    BEARISH_ENGULFING = "BEARISH_ENGULFING"
    HAMMER = "HAMMER"
    SHOOTING_STAR = "SHOOTING_STAR"
    DOJI = "DOJI"
    NONE = "NONE"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TradingSignal:
    """Representa um sinal de trading gerado"""
    timestamp: int
    symbol: str
    timeframe: str
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    confidence: float  # 0.0 a 1.0
    reason: str
    indicators: Dict
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'signal_type': self.signal_type.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'confidence': self.confidence,
            'reason': self.reason,
            'indicators': self.indicators
        }


# ============================================================================
# CLASSE PRINCIPAL: ScalpingStrategy
# ============================================================================

class ScalpingStrategy:
    """
    Estratégia de scalping com detecção de topos e fundos
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier_sl: float = 1.0,
        atr_multiplier_tp1: float = 1.0,
        atr_multiplier_tp2: float = 2.0,
        ema_fast: int = 9,
        ema_medium: int = 21,
        ema_slow: int = 50,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        swing_lookback: int = 5,
        min_atr_threshold: float = 0.0,
        confidence_threshold: float = 0.5  # ✅ Reduzido de 0.6 para 0.5
    ):
        """
        Inicializa a estratégia
        
        Args:
            atr_period: Período do ATR
            atr_multiplier_sl: Multiplicador ATR para Stop Loss
            atr_multiplier_tp1: Multiplicador ATR para Take Profit 1
            atr_multiplier_tp2: Multiplicador ATR para Take Profit 2
            ema_fast: Período da EMA rápida
            ema_medium: Período da EMA média
            ema_slow: Período da EMA lenta
            rsi_period: Período do RSI
            rsi_oversold: Nível de sobrevenda do RSI
            rsi_overbought: Nível de sobrecompra do RSI
            macd_fast/slow/signal: Parâmetros do MACD
            swing_lookback: Períodos para detectar swing high/low
            min_atr_threshold: ATR mínimo para filtrar baixa volatilidade
            confidence_threshold: Confiança mínima para gerar sinal (0-1)
        """
        # Parâmetros ATR e Risk
        self.atr_period = atr_period
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp1 = atr_multiplier_tp1
        self.atr_multiplier_tp2 = atr_multiplier_tp2
        self.min_atr_threshold = min_atr_threshold
        
        # Parâmetros EMA
        self.ema_fast = ema_fast
        self.ema_medium = ema_medium
        self.ema_slow = ema_slow
        
        # Parâmetros RSI
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # Parâmetros MACD
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
        # Swing detection
        self.swing_lookback = swing_lookback
        
        # Confidence threshold
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"ScalpingStrategy inicializada: {self.__dict__}")
    
    # ========================================================================
    # INDICADORES TÉCNICOS
    # ========================================================================
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todos os indicadores técnicos
        
        Args:
            df: DataFrame com colunas [open, high, low, close, volume]
        
        Returns:
            DataFrame com indicadores adicionados
        """
        if df.empty or len(df) < max(self.ema_slow, self.atr_period, self.macd_slow):
            logger.warning("DataFrame insuficiente para calcular indicadores")
            return df
        
        df = df.copy()
        
        # EMAs
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=self.ema_fast)
        df['ema_medium'] = ta.trend.ema_indicator(df['close'], window=self.ema_medium)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=self.ema_slow)
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
        
        # MACD
        macd = ta.trend.MACD(
            df['close'],
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(
            df['high'],
            df['low'],
            df['close'],
            window=self.atr_period
        )
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # Tendência EMA
        df['ema_trend'] = np.where(
            df['ema_fast'] > df['ema_medium'], 1,
            np.where(df['ema_fast'] < df['ema_medium'], -1, 0)
        )
        
        return df
    
    # ========================================================================
    # DETECÇÃO DE SWING HIGHS E LOWS
    # ========================================================================
    
    def detect_swing_high(self, df: pd.DataFrame, index: int) -> bool:
        """
        Detecta se há um swing high no índice especificado
        
        Um swing high ocorre quando o high no index é maior que
        os N highs anteriores e posteriores
        
        Args:
            df: DataFrame com dados
            index: Índice a verificar
        
        Returns:
            True se é swing high
        """
        if index < self.swing_lookback or index >= len(df) - self.swing_lookback:
            return False
        
        current_high = df.iloc[index]['high']
        
        # Verifica se é maior que lookback anteriores
        for i in range(1, self.swing_lookback + 1):
            if df.iloc[index - i]['high'] >= current_high:
                return False
        
        # Verifica se é maior que lookback posteriores
        for i in range(1, self.swing_lookback + 1):
            if df.iloc[index + i]['high'] >= current_high:
                return False
        
        return True
    
    def detect_swing_low(self, df: pd.DataFrame, index: int) -> bool:
        """
        Detecta se há um swing low no índice especificado
        
        Args:
            df: DataFrame com dados
            index: Índice a verificar
        
        Returns:
            True se é swing low
        """
        if index < self.swing_lookback or index >= len(df) - self.swing_lookback:
            return False
        
        current_low = df.iloc[index]['low']
        
        # Verifica se é menor que lookback anteriores
        for i in range(1, self.swing_lookback + 1):
            if df.iloc[index - i]['low'] <= current_low:
                return False
        
        # Verifica se é menor que lookback posteriores
        for i in range(1, self.swing_lookback + 1):
            if df.iloc[index + i]['low'] <= current_low:
                return False
        
        return True
    
    def find_recent_swing_points(self, df: pd.DataFrame) -> Dict:
        """
        Encontra os últimos swing highs e lows
        
        Returns:
            Dict com 'swing_high' e 'swing_low' (preço e index)
        """
        result = {
            'swing_high': None,
            'swing_low': None
        }
        
        # Busca do mais recente para o mais antigo
        for i in range(len(df) - self.swing_lookback - 1, self.swing_lookback, -1):
            if result['swing_high'] is None and self.detect_swing_high(df, i):
                result['swing_high'] = {
                    'price': df.iloc[i]['high'],
                    'index': i
                }
            
            if result['swing_low'] is None and self.detect_swing_low(df, i):
                result['swing_low'] = {
                    'price': df.iloc[i]['low'],
                    'index': i
                }
            
            # Para quando encontrar ambos
            if result['swing_high'] and result['swing_low']:
                break
        
        return result
    
    # ========================================================================
    # PADRÕES DE CANDLE
    # ========================================================================
    
    def detect_candle_pattern(self, df: pd.DataFrame, index: int = -1) -> CandlePattern:
        """
        Detecta padrões de candle
        
        Args:
            df: DataFrame com dados OHLC
            index: Índice do candle (-1 = último)
        
        Returns:
            CandlePattern detectado
        """
        if len(df) < 2:
            return CandlePattern.NONE
        
        current = df.iloc[index]
        previous = df.iloc[index - 1]
        
        # Bullish Engulfing
        if (previous['close'] < previous['open'] and  # Candle anterior bearish
            current['close'] > current['open'] and    # Candle atual bullish
            current['open'] < previous['close'] and   # Abre abaixo do close anterior
            current['close'] > previous['open']):     # Fecha acima do open anterior
            return CandlePattern.BULLISH_ENGULFING
        
        # Bearish Engulfing
        if (previous['close'] > previous['open'] and  # Candle anterior bullish
            current['close'] < current['open'] and    # Candle atual bearish
            current['open'] > previous['close'] and   # Abre acima do close anterior
            current['close'] < previous['open']):     # Fecha abaixo do open anterior
            return CandlePattern.BEARISH_ENGULFING
        
        # Hammer (bullish reversal)
        body = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        lower_wick = min(current['open'], current['close']) - current['low']
        upper_wick = current['high'] - max(current['open'], current['close'])
        
        if total_range > 0:
            if (lower_wick > body * 2 and           # Lower wick > 2x body
                upper_wick < body * 0.3 and         # Upper wick pequeno
                body / total_range < 0.3):          # Body pequeno
                return CandlePattern.HAMMER
        
        # Shooting Star (bearish reversal)
        if total_range > 0:
            if (upper_wick > body * 2 and           # Upper wick > 2x body
                lower_wick < body * 0.3 and         # Lower wick pequeno
                body / total_range < 0.3):          # Body pequeno
                return CandlePattern.SHOOTING_STAR
        
        # Doji (indecisão)
        if total_range > 0 and body / total_range < 0.1:
            return CandlePattern.DOJI
        
        return CandlePattern.NONE
    
    # ========================================================================
    # GERAÇÃO DE SINAIS - ✅ VERSÃO CORRIGIDA
    # ========================================================================
    
    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Optional[TradingSignal]:
        """
        Analisa o mercado e gera sinais de trading
        
        ✅ VERSÃO CORRIGIDA - Sistema de pontos mais generoso
        
        Args:
            df: DataFrame com dados OHLC
            symbol: Símbolo do ativo
            timeframe: Timeframe analisado
        
        Returns:
            TradingSignal ou None se não houver sinal
        """
        if df.empty or len(df) < 100:
            logger.debug(f"Dados insuficientes: {symbol} {timeframe}")
            return None
        
        # Calcula indicadores
        df = self.calculate_indicators(df)
        
        # Pega últimos valores
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Verifica volatilidade mínima
        if current['atr'] < self.min_atr_threshold:
            logger.debug(f"ATR muito baixo: {current['atr']:.2f}")
            return None
        
        # Detecta padrão de candle
        candle_pattern = self.detect_candle_pattern(df)
        
        # Encontra swing points
        swings = self.find_recent_swing_points(df)
        
        # ====================================================================
        # LÓGICA DE DETECÇÃO DE FUNDO (LONG) - ✅ CORRIGIDA
        # ====================================================================
        
        long_score = 0.0
        long_reasons = []
        
        # 1. RSI (20 pontos total)
        if current['rsi'] < self.rsi_oversold:
            long_score += 0.20
            long_reasons.append(f"RSI oversold ({current['rsi']:.1f})")
        elif current['rsi'] < 40:  # ✅ NOVO: Pontos parciais
            long_score += 0.10
            long_reasons.append(f"RSI baixo ({current['rsi']:.1f})")
        
        # 2. Padrão de candle (25 pontos total)
        if candle_pattern in [CandlePattern.BULLISH_ENGULFING, CandlePattern.HAMMER]:
            long_score += 0.25
            long_reasons.append(f"Pattern: {candle_pattern.value}")
        elif candle_pattern == CandlePattern.DOJI:  # ✅ NOVO
            long_score += 0.10
            long_reasons.append("Doji - indecisão")
        
        # 3. EMA (20 pontos total) - ✅ CORRIGIDO
        if previous['ema_fast'] <= previous['ema_medium'] and current['ema_fast'] > current['ema_medium']:
            long_score += 0.20
            long_reasons.append("EMA bullish crossover")
        elif current['ema_fast'] > current['ema_medium']:  # ✅ NOVO: Já em tendência
            long_score += 0.10
            long_reasons.append("EMA bullish trend")
        
        # 4. MACD (15 pontos total) - ✅ CORRIGIDO
        if previous['macd'] <= previous['macd_signal'] and current['macd'] > current['macd_signal']:
            long_score += 0.15
            long_reasons.append("MACD bullish cross")
        elif current['macd'] > current['macd_signal']:  # ✅ NOVO
            long_score += 0.08
            long_reasons.append("MACD positive")
        
        # 5. Swing low (15 pontos total)
        if swings['swing_low']:
            distance_to_low = abs(current['close'] - swings['swing_low']['price'])
            distance_pct = distance_to_low / current['close']
            if distance_pct < 0.01:  # Dentro de 1%
                long_score += 0.15
                long_reasons.append("Próximo de swing low")
            elif distance_pct < 0.02:  # ✅ NOVO: Dentro de 2%
                long_score += 0.08
                long_reasons.append("Perto de suporte")
        
        # 6. Bollinger Bands (10 pontos total)
        if current['close'] <= current['bb_lower']:
            long_score += 0.10
            long_reasons.append("Tocando BB inferior")
        elif current['close'] < current['bb_middle']:  # ✅ NOVO
            long_score += 0.05
            long_reasons.append("Abaixo de BB middle")
        
        # ====================================================================
        # LÓGICA DE DETECÇÃO DE TOPO (SHORT) - ✅ CORRIGIDA
        # ====================================================================
        
        short_score = 0.0
        short_reasons = []
        
        # 1. RSI (20 pontos total)
        if current['rsi'] > self.rsi_overbought:
            short_score += 0.20
            short_reasons.append(f"RSI overbought ({current['rsi']:.1f})")
        elif current['rsi'] > 60:  # ✅ NOVO
            short_score += 0.10
            short_reasons.append(f"RSI alto ({current['rsi']:.1f})")
        
        # 2. Padrão de candle (25 pontos total)
        if candle_pattern in [CandlePattern.BEARISH_ENGULFING, CandlePattern.SHOOTING_STAR]:
            short_score += 0.25
            short_reasons.append(f"Pattern: {candle_pattern.value}")
        elif candle_pattern == CandlePattern.DOJI:  # ✅ NOVO
            short_score += 0.10
            short_reasons.append("Doji - indecisão")
        
        # 3. EMA (20 pontos total) - ✅ CORRIGIDO
        if previous['ema_fast'] >= previous['ema_medium'] and current['ema_fast'] < current['ema_medium']:
            short_score += 0.20
            short_reasons.append("EMA bearish crossover")
        elif current['ema_fast'] < current['ema_medium']:  # ✅ NOVO
            short_score += 0.10
            short_reasons.append("EMA bearish trend")
        
        # 4. MACD (15 pontos total) - ✅ CORRIGIDO
        if previous['macd'] >= previous['macd_signal'] and current['macd'] < current['macd_signal']:
            short_score += 0.15
            short_reasons.append("MACD bearish cross")
        elif current['macd'] < current['macd_signal']:  # ✅ NOVO
            short_score += 0.08
            short_reasons.append("MACD negative")
        
        # 5. Swing high (15 pontos total)
        if swings['swing_high']:
            distance_to_high = abs(current['close'] - swings['swing_high']['price'])
            distance_pct = distance_to_high / current['close']
            if distance_pct < 0.01:
                short_score += 0.15
                short_reasons.append("Próximo de swing high")
            elif distance_pct < 0.02:  # ✅ NOVO
                short_score += 0.08
                short_reasons.append("Perto de resistência")
        
        # 6. Bollinger Bands (10 pontos total)
        if current['close'] >= current['bb_upper']:
            short_score += 0.10
            short_reasons.append("Tocando BB superior")
        elif current['close'] > current['bb_middle']:  # ✅ NOVO
            short_score += 0.05
            short_reasons.append("Acima de BB middle")
        
        # ====================================================================
        # DECISÃO FINAL - ✅ CORRIGIDA
        # ====================================================================
        
        signal_type = None
        confidence = 0.0
        reasons = []
        
        if long_score > short_score and long_score >= self.confidence_threshold:
            signal_type = SignalType.LONG
            confidence = min(long_score, 1.0)  # ✅ Cap em 1.0
            reasons = long_reasons
        elif short_score > long_score and short_score >= self.confidence_threshold:
            signal_type = SignalType.SHORT
            confidence = min(short_score, 1.0)
            reasons = short_reasons
        else:
            # ✅ NOVO: Log quando não atinge threshold
            logger.debug(
                f"{symbol} {timeframe}: Sem sinal forte - "
                f"LONG={long_score:.2f}, SHORT={short_score:.2f}, "
                f"threshold={self.confidence_threshold:.2f}"
            )
            return None
        
        # ====================================================================
        # CALCULA STOP LOSS E TAKE PROFITS
        # ====================================================================
        
        entry_price = current['close']
        atr = current['atr']
        
        if signal_type == SignalType.LONG:
            stop_loss = entry_price - (atr * self.atr_multiplier_sl)
            take_profit_1 = entry_price + (atr * self.atr_multiplier_tp1)
            take_profit_2 = entry_price + (atr * self.atr_multiplier_tp2)
        else:  # SHORT
            stop_loss = entry_price + (atr * self.atr_multiplier_sl)
            take_profit_1 = entry_price - (atr * self.atr_multiplier_tp1)
            take_profit_2 = entry_price - (atr * self.atr_multiplier_tp2)
        
        # Cria sinal
        signal = TradingSignal(
            timestamp=int(current.name.timestamp() * 1000) if hasattr(current.name, 'timestamp') else int(df.iloc[-1]['timestamp']),
            symbol=symbol,
            timeframe=timeframe,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            confidence=confidence,
            reason=" | ".join(reasons),
            indicators={
                'rsi': float(current['rsi']),
                'ema_fast': float(current['ema_fast']),
                'ema_medium': float(current['ema_medium']),
                'ema_slow': float(current['ema_slow']),
                'macd': float(current['macd']),
                'macd_signal': float(current['macd_signal']),
                'atr': float(current['atr']),
                'candle_pattern': candle_pattern.value,
                'long_score': long_score,
                'short_score': short_score
            }
        )
        
        logger.info(
            f"🎯 SINAL GERADO: {signal.signal_type.value} {symbol} {timeframe} | "
            f"Confidence: {confidence:.2%} | Entry: {entry_price:.2f} | "
            f"SL: {stop_loss:.2f} | TP1: {take_profit_1:.2f} | TP2: {take_profit_2:.2f}"
        )
        logger.info(f"   Razões: {signal.reason}")
        
        return signal
    
    # ========================================================================
    # CONFIRMAÇÃO MULTI-TIMEFRAME
    # ========================================================================
    
    def confirm_signal_multi_tf(
        self,
        signal_primary: TradingSignal,
        df_higher_tf: pd.DataFrame,
        higher_tf_name: str
    ) -> bool:
        """
        Confirma sinal do timeframe menor com timeframe maior
        
        Args:
            signal_primary: Sinal do timeframe primário (ex: 15m)
            df_higher_tf: DataFrame do timeframe maior (ex: 1h)
            higher_tf_name: Nome do timeframe maior
        
        Returns:
            True se confirmado, False caso contrário
        """
        if df_higher_tf.empty or len(df_higher_tf) < 50:
            logger.warning(f"Dados insuficientes no TF superior: {higher_tf_name}")
            return False
        
        # Calcula indicadores no TF superior
        df_higher = self.calculate_indicators(df_higher_tf)
        current = df_higher.iloc[-1]
        
        # Para LONG, confirmamos se:
        if signal_primary.signal_type == SignalType.LONG:
            # 1. EMA fast > medium no TF superior (tendência bullish)
            ema_confirm = current['ema_fast'] > current['ema_medium']
            
            # 2. RSI não está em extremo de sobrecompra
            rsi_confirm = current['rsi'] < 75
            
            # 3. MACD está positivo ou cruzando pra cima
            macd_confirm = current['macd'] > current['macd_signal'] or current['macd_diff'] > 0
            
            confirmed = ema_confirm and rsi_confirm
            
            if confirmed:
                logger.info(
                    f"✅ Confirmação LONG no {higher_tf_name}: "
                    f"EMA={ema_confirm}, RSI={rsi_confirm}, MACD={macd_confirm}"
                )
            else:
                logger.warning(
                    f"❌ LONG não confirmado no {higher_tf_name}: "
                    f"EMA={ema_confirm}, RSI={rsi_confirm}"
                )
            
            return confirmed
        
        # Para SHORT, confirmamos se:
        elif signal_primary.signal_type == SignalType.SHORT:
            # 1. EMA fast < medium no TF superior (tendência bearish)
            ema_confirm = current['ema_fast'] < current['ema_medium']
            
            # 2. RSI não está em extremo de sobrevenda
            rsi_confirm = current['rsi'] > 25
            
            # 3. MACD está negativo ou cruzando pra baixo
            macd_confirm = current['macd'] < current['macd_signal'] or current['macd_diff'] < 0
            
            confirmed = ema_confirm and rsi_confirm
            
            if confirmed:
                logger.info(
                    f"✅ Confirmação SHORT no {higher_tf_name}: "
                    f"EMA={ema_confirm}, RSI={rsi_confirm}, MACD={macd_confirm}"
                )
            else:
                logger.warning(
                    f"❌ SHORT não confirmado no {higher_tf_name}: "
                    f"EMA={ema_confirm}, RSI={rsi_confirm}"
                )
            
            return confirmed
        
        return False
    
    # ========================================================================
    # UTILITÁRIOS
    # ========================================================================
    
    def should_move_sl_to_breakeven(
        self,
        entry_price: float,
        current_price: float,
        tp1_price: float,
        signal_type: SignalType
    ) -> bool:
        """
        Verifica se deve mover SL para breakeven
        
        Returns:
            True se TP1 foi atingido e SL deve ir para breakeven
        """
        if signal_type == SignalType.LONG:
            return current_price >= tp1_price
        elif signal_type == SignalType.SHORT:
            return current_price <= tp1_price
        
        return False
    
    def __repr__(self) -> str:
        return (
            f"ScalpingStrategy("
            f"ATR={self.atr_period}, "
            f"EMA={self.ema_fast}/{self.ema_medium}/{self.ema_slow}, "
            f"RSI={self.rsi_period}, "
            f"confidence_min={self.confidence_threshold})"
        )


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Exemplo de uso da estratégia
    """
    
    # Configuração de log
    logger.add(
        "logs/strategy.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )
    
    # Dados fictícios para teste
    dates = pd.date_range('2024-01-01', periods=200, freq='15min')
    np.random.seed(42)
    
    # Simula preços
    close_prices = 42000 + np.cumsum(np.random.randn(200) * 50)
    high_prices = close_prices + np.random.rand(200) * 100
    low_prices = close_prices - np.random.rand(200) * 100
    open_prices = np.roll(close_prices, 1)
    volumes = np.random.rand(200) * 1000
    
    df = pd.DataFrame({
        'timestamp': [int(d.timestamp() * 1000) for d in dates],
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    df.index = dates
    
    # Inicializa estratégia
    strategy = ScalpingStrategy(
        atr_multiplier_sl=1.0,
        atr_multiplier_tp1=1.0,
        atr_multiplier_tp2=2.0,
        confidence_threshold=0.5  # ✅ Threshold reduzido
    )
    
    # Analisa
    signal = strategy.analyze(df, "BTCUSDT", "15")
    
    if signal:
        logger.success(f"Sinal gerado: {signal.to_dict()}")
    else:
        logger.info("Nenhum sinal gerado")
    
    # Teste de confirmação multi-timeframe
    df_1h = df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'timestamp': 'first'
    }).dropna()
    
    if signal:
        confirmed = strategy.confirm_signal_multi_tf(signal, df_1h, "1H")
        logger.info(f"Confirmação 1H: {confirmed}")
