"""
Dynamic Parameter Adjustment - Ajuste Dinâmico de Parâmetros
============================================================

Módulo para ajustar parâmetros da estratégia automaticamente baseado em:
- Volatilidade do mercado (ATR, realized volatility)
- Regime de mercado (trending, ranging, choppy)
- Volume
- Spread bid-ask
- Condições de liquidez

Adapta automaticamente:
- ATR multipliers (SL/TP)
- Risk per trade
- Confidence threshold
- Position sizing
- Trailing stop distance

Autor: Trading Bot Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


# ============================================================================
# ENUMS E CONSTANTES
# ============================================================================

class VolatilityRegime(Enum):
    """Regimes de volatilidade"""
    VERY_LOW = "VERY_LOW"      # <20 percentile
    LOW = "LOW"                # 20-40 percentile
    MEDIUM = "MEDIUM"          # 40-60 percentile
    HIGH = "HIGH"              # 60-80 percentile
    VERY_HIGH = "VERY_HIGH"    # >80 percentile


class MarketRegime(Enum):
    """Regimes de mercado"""
    STRONG_TRENDING = "STRONG_TRENDING"
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"


class LiquidityRegime(Enum):
    """Regimes de liquidez"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MarketConditions:
    """Condições atuais do mercado"""
    timestamp: datetime
    
    # Volatilidade
    atr: float
    atr_percentile: float
    realized_volatility: float
    volatility_regime: VolatilityRegime
    
    # Tendência
    adx: float
    trend_strength: float
    market_regime: MarketRegime
    
    # Volume e liquidez
    volume: float
    volume_ma: float
    volume_ratio: float
    liquidity_regime: LiquidityRegime
    
    # Spread
    spread_pct: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'atr': self.atr,
            'atr_percentile': self.atr_percentile,
            'realized_volatility': self.realized_volatility,
            'volatility_regime': self.volatility_regime.value,
            'adx': self.adx,
            'trend_strength': self.trend_strength,
            'market_regime': self.market_regime.value,
            'volume': self.volume,
            'volume_ratio': self.volume_ratio,
            'liquidity_regime': self.liquidity_regime.value
        }


@dataclass
class DynamicParameters:
    """Parâmetros ajustados dinamicamente"""
    # ATR Multipliers
    atr_multiplier_sl: float
    atr_multiplier_tp1: float
    atr_multiplier_tp2: float
    
    # Risk management
    risk_per_trade: float
    max_positions: int
    confidence_threshold: float
    
    # Trailing
    trailing_sl_enabled: bool
    trailing_sl_pct: float
    
    # Execution
    partial_tp1_pct: float
    
    # Reasoning
    adjustments_made: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'atr_multiplier_sl': self.atr_multiplier_sl,
            'atr_multiplier_tp1': self.atr_multiplier_tp1,
            'atr_multiplier_tp2': self.atr_multiplier_tp2,
            'risk_per_trade': self.risk_per_trade,
            'max_positions': self.max_positions,
            'confidence_threshold': self.confidence_threshold,
            'trailing_sl_enabled': self.trailing_sl_enabled,
            'trailing_sl_pct': self.trailing_sl_pct,
            'partial_tp1_pct': self.partial_tp1_pct,
            'adjustments_made': self.adjustments_made
        }


# ============================================================================
# CLASSE PRINCIPAL: DynamicParameterManager
# ============================================================================

class DynamicParameterManager:
    """
    Gerenciador de ajuste dinâmico de parâmetros
    """
    
    def __init__(
        self,
        # Parâmetros base (serão ajustados dinamicamente)
        base_atr_multiplier_sl: float = 1.0,
        base_atr_multiplier_tp1: float = 1.0,
        base_atr_multiplier_tp2: float = 2.0,
        base_risk_per_trade: float = 0.01,
        base_confidence_threshold: float = 0.6,
        base_max_positions: int = 2,
        
        # Configurações de ajuste
        volatility_lookback: int = 100,
        trend_lookback: int = 50,
        enable_volatility_adjustment: bool = True,
        enable_trend_adjustment: bool = True,
        enable_liquidity_adjustment: bool = True,
        
        # Limites de ajuste
        max_sl_multiplier: float = 2.0,
        min_sl_multiplier: float = 0.5,
        max_risk_adjustment: float = 2.0,
        min_risk_adjustment: float = 0.5
    ):
        """
        Inicializa o gerenciador
        
        Args:
            base_*: Parâmetros base a serem ajustados
            *_lookback: Períodos para cálculo de métricas
            enable_*_adjustment: Flags para habilitar ajustes
            max/min_*: Limites de ajuste
        """
        # Parâmetros base
        self.base_atr_multiplier_sl = base_atr_multiplier_sl
        self.base_atr_multiplier_tp1 = base_atr_multiplier_tp1
        self.base_atr_multiplier_tp2 = base_atr_multiplier_tp2
        self.base_risk_per_trade = base_risk_per_trade
        self.base_confidence_threshold = base_confidence_threshold
        self.base_max_positions = base_max_positions
        
        # Configurações
        self.volatility_lookback = volatility_lookback
        self.trend_lookback = trend_lookback
        self.enable_volatility_adjustment = enable_volatility_adjustment
        self.enable_trend_adjustment = enable_trend_adjustment
        self.enable_liquidity_adjustment = enable_liquidity_adjustment
        
        # Limites
        self.max_sl_multiplier = max_sl_multiplier
        self.min_sl_multiplier = min_sl_multiplier
        self.max_risk_adjustment = max_risk_adjustment
        self.min_risk_adjustment = min_risk_adjustment
        
        # Cache de ATR histórico para percentis
        self.atr_history = []
        
        # Parâmetros atuais
        self.current_params: Optional[DynamicParameters] = None
        self.current_conditions: Optional[MarketConditions] = None
        
        logger.info(
            f"DynamicParameterManager inicializado: "
            f"vol_adj={enable_volatility_adjustment}, "
            f"trend_adj={enable_trend_adjustment}, "
            f"liq_adj={enable_liquidity_adjustment}"
        )
    
    # ========================================================================
    # ANÁLISE DE CONDIÇÕES DE MERCADO
    # ========================================================================
    
    def analyze_market_conditions(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> MarketConditions:
        """
        Analisa condições atuais do mercado
        
        Args:
            df: DataFrame com OHLCV e indicadores
            current_price: Preço atual (opcional)
        
        Returns:
            MarketConditions
        """
        if df.empty or len(df) < self.volatility_lookback:
            raise ValueError("DataFrame insuficiente para análise")
        
        # Usa últimos dados
        recent_df = df.tail(self.volatility_lookback)
        latest = recent_df.iloc[-1]
        
        # ====================================================================
        # 1. VOLATILIDADE
        # ====================================================================
        
        # ATR
        if 'atr' in recent_df.columns:
            atr = latest['atr']
        else:
            # Calcula ATR se não existir
            high_low = recent_df['high'] - recent_df['low']
            high_close = abs(recent_df['high'] - recent_df['close'].shift(1))
            low_close = abs(recent_df['low'] - recent_df['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
        
        # ATR percentile
        self.atr_history.append(atr)
        if len(self.atr_history) > 500:  # Mantém últimos 500
            self.atr_history.pop(0)
        
        if len(self.atr_history) > 20:
            atr_percentile = (
                sum(1 for x in self.atr_history if x < atr) / len(self.atr_history)
            )
        else:
            atr_percentile = 0.5
        
        # Volatilidade realizada
        returns = recent_df['close'].pct_change()
        realized_vol = returns.std() * np.sqrt(252 * 24 * 60 / 15)  # Anualizada
        
        # Classifica regime de volatilidade
        if atr_percentile < 0.2:
            vol_regime = VolatilityRegime.VERY_LOW
        elif atr_percentile < 0.4:
            vol_regime = VolatilityRegime.LOW
        elif atr_percentile < 0.6:
            vol_regime = VolatilityRegime.MEDIUM
        elif atr_percentile < 0.8:
            vol_regime = VolatilityRegime.HIGH
        else:
            vol_regime = VolatilityRegime.VERY_HIGH
        
        # ====================================================================
        # 2. TENDÊNCIA E REGIME DE MERCADO
        # ====================================================================
        
        # Calcula ADX se não existir
        if 'adx' in recent_df.columns:
            adx = latest['adx']
        else:
            adx = self._calculate_adx(recent_df)
        
        # Trend strength (baseado em EMA)
        if 'ema_fast' in recent_df.columns and 'ema_slow' in recent_df.columns:
            ema_distance = abs(
                (latest['ema_fast'] - latest['ema_slow']) / latest['close']
            )
            trend_strength = min(ema_distance * 100, 1.0)
        else:
            trend_strength = 0.5
        
        # Classifica regime de mercado
        if adx > 40 and trend_strength > 0.02:
            market_regime = MarketRegime.STRONG_TRENDING
        elif adx > 25 and trend_strength > 0.01:
            market_regime = MarketRegime.TRENDING
        elif adx < 20:
            market_regime = MarketRegime.RANGING
        else:
            market_regime = MarketRegime.CHOPPY
        
        # ====================================================================
        # 3. VOLUME E LIQUIDEZ
        # ====================================================================
        
        volume = latest['volume']
        volume_ma = recent_df['volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = volume / volume_ma if volume_ma > 0 else 1.0
        
        # Classifica liquidez
        if volume_ratio > 1.5:
            liquidity_regime = LiquidityRegime.HIGH
        elif volume_ratio > 0.7:
            liquidity_regime = LiquidityRegime.MEDIUM
        else:
            liquidity_regime = LiquidityRegime.LOW
        
        # ====================================================================
        # Cria objeto de condições
        # ====================================================================
        
        conditions = MarketConditions(
            timestamp=latest.name if hasattr(latest.name, 'to_pydatetime') else datetime.now(),
            atr=atr,
            atr_percentile=atr_percentile,
            realized_volatility=realized_vol,
            volatility_regime=vol_regime,
            adx=adx,
            trend_strength=trend_strength,
            market_regime=market_regime,
            volume=volume,
            volume_ma=volume_ma,
            volume_ratio=volume_ratio,
            liquidity_regime=liquidity_regime
        )
        
        self.current_conditions = conditions
        
        return conditions
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calcula ADX (Average Directional Index)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1] if not adx.empty else 25.0
    
    # ========================================================================
    # AJUSTE DE PARÂMETROS
    # ========================================================================
    
    def adjust_parameters(
        self,
        conditions: Optional[MarketConditions] = None,
        df: Optional[pd.DataFrame] = None
    ) -> DynamicParameters:
        """
        Ajusta parâmetros baseado nas condições de mercado
        
        Args:
            conditions: Condições de mercado (ou None para calcular)
            df: DataFrame para calcular condições (se conditions=None)
        
        Returns:
            DynamicParameters ajustados
        """
        # Analisa condições se não fornecidas
        if conditions is None:
            if df is None:
                raise ValueError("Forneça conditions ou df")
            conditions = self.analyze_market_conditions(df)
        
        # Inicia com parâmetros base
        params = DynamicParameters(
            atr_multiplier_sl=self.base_atr_multiplier_sl,
            atr_multiplier_tp1=self.base_atr_multiplier_tp1,
            atr_multiplier_tp2=self.base_atr_multiplier_tp2,
            risk_per_trade=self.base_risk_per_trade,
            max_positions=self.base_max_positions,
            confidence_threshold=self.base_confidence_threshold,
            trailing_sl_enabled=False,
            trailing_sl_pct=0.005,
            partial_tp1_pct=0.5
        )
        
        adjustments = []
        
        # ====================================================================
        # 1. AJUSTES POR VOLATILIDADE
        # ====================================================================
        
        if self.enable_volatility_adjustment:
            vol_regime = conditions.volatility_regime
            
            if vol_regime == VolatilityRegime.VERY_LOW:
                # Baixa volatilidade: SL/TP mais apertados
                params.atr_multiplier_sl *= 0.7
                params.atr_multiplier_tp1 *= 0.7
                params.atr_multiplier_tp2 *= 0.8
                params.risk_per_trade *= 1.2  # Pode arriscar um pouco mais
                adjustments.append("Vol MUITO BAIXA: SL/TP reduzidos, risco aumentado")
            
            elif vol_regime == VolatilityRegime.LOW:
                params.atr_multiplier_sl *= 0.85
                params.atr_multiplier_tp1 *= 0.85
                params.atr_multiplier_tp2 *= 0.9
                params.risk_per_trade *= 1.1
                adjustments.append("Vol BAIXA: SL/TP levemente reduzidos")
            
            elif vol_regime == VolatilityRegime.HIGH:
                # Alta volatilidade: SL/TP mais largos
                params.atr_multiplier_sl *= 1.2
                params.atr_multiplier_tp1 *= 1.2
                params.atr_multiplier_tp2 *= 1.3
                params.risk_per_trade *= 0.9  # Reduz risco
                adjustments.append("Vol ALTA: SL/TP aumentados, risco reduzido")
            
            elif vol_regime == VolatilityRegime.VERY_HIGH:
                params.atr_multiplier_sl *= 1.5
                params.atr_multiplier_tp1 *= 1.3
                params.atr_multiplier_tp2 *= 1.5
                params.risk_per_trade *= 0.7
                params.confidence_threshold *= 1.1  # Mais seletivo
                adjustments.append("Vol MUITO ALTA: SL/TP ampliados, risco reduzido, mais seletivo")
        
        # ====================================================================
        # 2. AJUSTES POR REGIME DE MERCADO
        # ====================================================================
        
        if self.enable_trend_adjustment:
            market_regime = conditions.market_regime
            
            if market_regime == MarketRegime.STRONG_TRENDING:
                # Trending forte: trailing stop + TPs mais distantes
                params.atr_multiplier_tp1 *= 1.2
                params.atr_multiplier_tp2 *= 1.5
                params.trailing_sl_enabled = True
                params.partial_tp1_pct = 0.3  # Mantém mais na posição
                adjustments.append("STRONG TREND: TPs distantes, trailing SL ativo, mantém posição")
            
            elif market_regime == MarketRegime.TRENDING:
                params.atr_multiplier_tp2 *= 1.2
                params.trailing_sl_enabled = True
                adjustments.append("TREND: TP2 aumentado, trailing SL ativo")
            
            elif market_regime == MarketRegime.RANGING:
                # Ranging: TPs mais próximos, mais seletivo
                params.atr_multiplier_tp1 *= 0.8
                params.atr_multiplier_tp2 *= 0.9
                params.confidence_threshold *= 1.15
                params.partial_tp1_pct = 0.6  # Fecha mais cedo
                adjustments.append("RANGING: TPs reduzidos, mais seletivo, fecha mais cedo")
            
            elif market_regime == MarketRegime.CHOPPY:
                # Choppy: muito conservador
                params.atr_multiplier_sl *= 1.1  # SL mais largo
                params.atr_multiplier_tp1 *= 0.7  # TP muito próximo
                params.confidence_threshold *= 1.2
                params.risk_per_trade *= 0.8
                adjustments.append("CHOPPY: Conservador, TPs curtos, mais seletivo")
        
        # ====================================================================
        # 3. AJUSTES POR LIQUIDEZ
        # ====================================================================
        
        if self.enable_liquidity_adjustment:
            liquidity = conditions.liquidity_regime
            
            if liquidity == LiquidityRegime.LOW:
                # Baixa liquidez: reduz posições e aumenta slippage expectations
                params.max_positions = max(1, self.base_max_positions - 1)
                params.risk_per_trade *= 0.9
                adjustments.append("Liquidez BAIXA: reduz posições e risco")
            
            elif liquidity == LiquidityRegime.HIGH:
                # Alta liquidez: pode aumentar posições
                params.max_positions = self.base_max_positions + 1
                adjustments.append("Liquidez ALTA: permite mais posições")
        
        # ====================================================================
        # 4. APLICA LIMITES
        # ====================================================================
        
        params.atr_multiplier_sl = np.clip(
            params.atr_multiplier_sl,
            self.min_sl_multiplier,
            self.max_sl_multiplier
        )
        
        params.atr_multiplier_tp1 = np.clip(
            params.atr_multiplier_tp1,
            self.min_sl_multiplier,
            self.max_sl_multiplier
        )
        
        params.atr_multiplier_tp2 = np.clip(
            params.atr_multiplier_tp2,
            self.min_sl_multiplier * 1.5,
            self.max_sl_multiplier * 2
        )
        
        params.risk_per_trade = np.clip(
            params.risk_per_trade,
            self.base_risk_per_trade * self.min_risk_adjustment,
            self.base_risk_per_trade * self.max_risk_adjustment
        )
        
        params.confidence_threshold = np.clip(
            params.confidence_threshold,
            0.5,
            0.9
        )
        
        params.adjustments_made = adjustments
        
        # Armazena parâmetros atuais
        self.current_params = params
        
        # Log
        if adjustments:
            logger.info(f"🔧 Parâmetros ajustados dinamicamente:")
            logger.info(f"   Condições: {conditions.volatility_regime.value} / {conditions.market_regime.value}")
            for adj in adjustments:
                logger.info(f"   • {adj}")
            logger.info(f"   SL: {params.atr_multiplier_sl:.2f}x | TP1: {params.atr_multiplier_tp1:.2f}x | TP2: {params.atr_multiplier_tp2:.2f}x")
            logger.info(f"   Risk: {params.risk_per_trade*100:.2f}% | Confidence: {params.confidence_threshold:.1%}")
        
        return params
    
    # ========================================================================
    # UTILITÁRIOS
    # ========================================================================
    
    def get_current_parameters(self) -> Optional[DynamicParameters]:
        """Retorna parâmetros atuais"""
        return self.current_params
    
    def get_current_conditions(self) -> Optional[MarketConditions]:
        """Retorna condições atuais"""
        return self.current_conditions
    
    def reset(self):
        """Reseta para parâmetros base"""
        self.current_params = None
        self.current_conditions = None
        self.atr_history = []
        logger.info("Parâmetros resetados para valores base")
    
    def get_adjustment_summary(self) -> Dict:
        """Retorna resumo dos ajustes atuais"""
        if not self.current_params or not self.current_conditions:
            return {}
        
        return {
            'conditions': self.current_conditions.to_dict(),
            'parameters': self.current_params.to_dict(),
            'adjustments': self.current_params.adjustments_made
        }


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Exemplo de uso do Dynamic Parameter Manager
    """
    from loguru import logger
    
    logger.add("logs/dynamic_params.log", rotation="1 day")
    
    # Dados fictícios
    dates = pd.date_range('2024-01-01', periods=500, freq='15min')
    np.random.seed(42)
    
    close_prices = 42000 + np.cumsum(np.random.randn(500) * 50)
    
    df = pd.DataFrame({
        'open': close_prices - np.random.rand(500) * 20,
        'high': close_prices + np.random.rand(500) * 50,
        'low': close_prices - np.random.rand(500) * 50,
        'close': close_prices,
        'volume': np.random.rand(500) * 1000
    }, index=dates)
    
    # Adiciona indicadores simulados
    df['ema_fast'] = df['close'].ewm(span=9).mean()
    df['ema_slow'] = df['close'].ewm(span=50).mean()
    df['atr'] = 100 + np.random.rand(500) * 50
    
    # Cria gerenciador
    manager = DynamicParameterManager(
        base_atr_multiplier_sl=1.0,
        base_atr_multiplier_tp1=1.0,
        base_atr_multiplier_tp2=2.0,
        base_risk_per_trade=0.01,
        enable_volatility_adjustment=True,
        enable_trend_adjustment=True,
        enable_liquidity_adjustment=True
    )
    
    # Analisa condições
    conditions = manager.analyze_market_conditions(df)
    
    logger.info("\n📊 Condições de Mercado:")
    logger.info(f"   Volatilidade: {conditions.volatility_regime.value}")
    logger.info(f"   ATR Percentil: {conditions.atr_percentile:.1%}")
    logger.info(f"   Regime: {conditions.market_regime.value}")
    logger.info(f"   ADX: {conditions.adx:.1f}")
    logger.info(f"   Liquidez: {conditions.liquidity_regime.value}")
    
    # Ajusta parâmetros
    params = manager.adjust_parameters(conditions)
    
    logger.info("\n🔧 Parâmetros Ajustados:")
    logger.info(f"   ATR SL: {params.atr_multiplier_sl:.2f}x")
    logger.info(f"   ATR TP1: {params.atr_multiplier_tp1:.2f}x")
    logger.info(f"   ATR TP2: {params.atr_multiplier_tp2:.2f}x")
    logger.info(f"   Risk: {params.risk_per_trade*100:.2f}%")
    logger.info(f"   Confidence: {params.confidence_threshold:.1%}")
    logger.info(f"   Max Pos: {params.max_positions}")
    logger.info(f"   Trailing: {params.trailing_sl_enabled}")
    
    logger.info("\n✅ Ajustes aplicados:")
    for adj in params.adjustments_made:
        logger.info(f"   • {adj}")
    
    # Testa com diferentes condições
    logger.info("\n🔄 Simulando diferentes condições:")
    
    # Alta volatilidade
    df['atr'] = 200 + np.random.rand(500) * 100  # Aumenta ATR
    conditions_high_vol = manager.analyze_market_conditions(df)
    params_high_vol = manager.adjust_parameters(conditions_high_vol)
    
    logger.info(f"\n   Alta Vol: SL={params_high_vol.atr_multiplier_sl:.2f}x, Risk={params_high_vol.risk_per_trade*100:.2f}%")
    
    # Baixa volatilidade
    df['atr'] = 20 + np.random.rand(500) * 10  # Diminui ATR
    conditions_low_vol = manager.analyze_market_conditions(df)
    params_low_vol = manager.adjust_parameters(conditions_low_vol)
    
    logger.info(f"   Baixa Vol: SL={params_low_vol.atr_multiplier_sl:.2f}x, Risk={params_low_vol.risk_per_trade*100:.2f}%")