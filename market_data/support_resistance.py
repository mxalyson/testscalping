"""
Support & Resistance Detector - Detecção Automática de Níveis
=============================================================

Módulo para detecção automática de níveis de suporte e resistência usando:
- Clustering de swing highs/lows (K-Means, DBSCAN)
- Volume Profile (POC, VAH, VAL)
- Pivot Points (Standard, Fibonacci, Camarilla)
- Fibonacci Retracements
- Zonas de consolidação
- Níveis psicológicos (números redondos)

Features:
- Detecção automática de níveis importantes
- Força de cada nível (baseado em toques, volume)
- Classificação por relevância
- Zonas (não apenas linhas)
- Validação temporal

Autor: Trading Bot Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


# ============================================================================
# ENUMS E CONSTANTES
# ============================================================================

class LevelType(Enum):
    """Tipos de níveis"""
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"
    PIVOT = "PIVOT"


class LevelStrength(Enum):
    """Força do nível"""
    WEAK = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


class LevelOrigin(Enum):
    """Origem do nível"""
    SWING_POINT = "SWING_POINT"
    VOLUME_PROFILE = "VOLUME_PROFILE"
    PIVOT_POINT = "PIVOT_POINT"
    FIBONACCI = "FIBONACCI"
    PSYCHOLOGICAL = "PSYCHOLOGICAL"
    CONSOLIDATION = "CONSOLIDATION"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SupportResistanceLevel:
    """Representa um nível de suporte ou resistência"""
    price: float
    level_type: LevelType
    strength: LevelStrength
    origin: LevelOrigin
    
    # Zona (min/max)
    zone_min: float
    zone_max: float
    
    # Métricas
    touches: int = 0
    volume_at_level: float = 0.0
    last_touch_time: Optional[datetime] = None
    age_candles: int = 0
    
    # Score
    relevance_score: float = 0.0
    
    def __post_init__(self):
        """Calcula zona se não fornecida"""
        if self.zone_min == 0:
            self.zone_min = self.price * 0.998  # 0.2% abaixo
        if self.zone_max == 0:
            self.zone_max = self.price * 1.002  # 0.2% acima
    
    def is_price_in_zone(self, price: float, tolerance: float = 0.002) -> bool:
        """Verifica se preço está na zona do nível"""
        zone_range = self.price * tolerance
        return (self.price - zone_range) <= price <= (self.price + zone_range)
    
    def distance_from_price(self, price: float) -> float:
        """Distância percentual do preço até o nível"""
        return abs(price - self.price) / price
    
    def to_dict(self) -> Dict:
        return {
            'price': self.price,
            'level_type': self.level_type.value,
            'strength': self.strength.value,
            'origin': self.origin.value,
            'zone_min': self.zone_min,
            'zone_max': self.zone_max,
            'touches': self.touches,
            'volume_at_level': self.volume_at_level,
            'last_touch_time': self.last_touch_time.isoformat() if self.last_touch_time else None,
            'age_candles': self.age_candles,
            'relevance_score': self.relevance_score
        }


@dataclass
class SRAnalysis:
    """Análise completa de S/R"""
    support_levels: List[SupportResistanceLevel] = field(default_factory=list)
    resistance_levels: List[SupportResistanceLevel] = field(default_factory=list)
    current_price: float = 0.0
    
    # Níveis mais próximos
    nearest_support: Optional[SupportResistanceLevel] = None
    nearest_resistance: Optional[SupportResistanceLevel] = None
    
    # Zona atual
    in_support_zone: bool = False
    in_resistance_zone: bool = False
    in_consolidation: bool = False
    
    def get_all_levels(self) -> List[SupportResistanceLevel]:
        """Retorna todos os níveis ordenados por relevância"""
        all_levels = self.support_levels + self.resistance_levels
        return sorted(all_levels, key=lambda x: x.relevance_score, reverse=True)
    
    def to_dict(self) -> Dict:
        return {
            'current_price': self.current_price,
            'support_levels': [s.to_dict() for s in self.support_levels],
            'resistance_levels': [r.to_dict() for r in self.resistance_levels],
            'nearest_support': self.nearest_support.to_dict() if self.nearest_support else None,
            'nearest_resistance': self.nearest_resistance.to_dict() if self.nearest_resistance else None,
            'in_support_zone': self.in_support_zone,
            'in_resistance_zone': self.in_resistance_zone,
            'in_consolidation': self.in_consolidation
        }


# ============================================================================
# CLASSE PRINCIPAL: SupportResistanceDetector
# ============================================================================

class SupportResistanceDetector:
    """
    Detector automático de suporte e resistência
    """
    
    def __init__(
        self,
        swing_window: int = 5,
        cluster_threshold: float = 0.002,
        min_touches: int = 2,
        min_strength: float = 0.3,
        zone_width_pct: float = 0.003,
        lookback_periods: int = 200,
        recency_weight: float = 0.3,
        volume_weight: float = 0.3
    ):
        """
        Inicializa o detector
        
        Args:
            swing_window: Janela para detectar swing highs/lows
            cluster_threshold: Threshold para agrupar preços próximos (%)
            min_touches: Mínimo de toques para considerar válido
            min_strength: Força mínima para incluir (0-1)
            zone_width_pct: Largura da zona S/R (%)
            lookback_periods: Períodos a analisar
            recency_weight: Peso para níveis recentes (0-1)
            volume_weight: Peso do volume no cálculo de força (0-1)
        """
        self.swing_window = swing_window
        self.cluster_threshold = cluster_threshold
        self.min_touches = min_touches
        self.min_strength = min_strength
        self.zone_width_pct = zone_width_pct
        self.lookback_periods = lookback_periods
        self.recency_weight = recency_weight
        self.volume_weight = volume_weight
        
        logger.info(
            f"SupportResistanceDetector inicializado: "
            f"swing_window={swing_window}, cluster_threshold={cluster_threshold*100:.2f}%"
        )
    
    # ========================================================================
    # ANÁLISE PRINCIPAL
    # ========================================================================
    
    def analyze(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> SRAnalysis:
        """
        Analisa e detecta níveis de S/R
        
        Args:
            df: DataFrame com OHLCV
            current_price: Preço atual (opcional)
        
        Returns:
            SRAnalysis com níveis detectados
        """
        if df.empty or len(df) < self.swing_lookback * 2:
            logger.warning("Dados insuficientes para análise S/R")
            return SRAnalysis()
        
        # Usa últimos dados
        df_recent = df.tail(self.lookback_periods).copy()
        
        # Preço atual
        if current_price is None:
            current_price = df_recent.iloc[-1]['close']
        
        logger.info(f"🔍 Detectando S/R para preço atual: {current_price:.2f}")
        
        # Lista de níveis encontrados
        all_levels: List[SupportResistanceLevel] = []
        
        # ====================================================================
        # 1. SWING HIGHS/LOWS COM CLUSTERING
        # ====================================================================
        
        swing_levels = self._detect_swing_levels(df_recent, current_price)
        all_levels.extend(swing_levels)
        logger.debug(f"   Swing levels: {len(swing_levels)}")
        
        # ====================================================================
        # 2. VOLUME PROFILE
        # ====================================================================
        
        if self.enable_volume_profile:
            volume_levels = self._detect_volume_levels(df_recent, current_price)
            all_levels.extend(volume_levels)
            logger.debug(f"   Volume levels: {len(volume_levels)}")
        
        # ====================================================================
        # 3. PIVOT POINTS
        # ====================================================================
        
        if self.enable_pivot_points:
            pivot_levels = self._calculate_pivot_points(df_recent, current_price)
            all_levels.extend(pivot_levels)
            logger.debug(f"   Pivot levels: {len(pivot_levels)}")
        
        # ====================================================================
        # 4. FIBONACCI RETRACEMENTS
        # ====================================================================
        
        if self.enable_fibonacci:
            fib_levels = self._calculate_fibonacci_levels(df_recent, current_price)
            all_levels.extend(fib_levels)
            logger.debug(f"   Fibonacci levels: {len(fib_levels)}")
        
        # ====================================================================
        # 5. NÍVEIS PSICOLÓGICOS
        # ====================================================================
        
        if self.enable_psychological:
            psych_levels = self._detect_psychological_levels(current_price)
            all_levels.extend(psych_levels)
            logger.debug(f"   Psychological levels: {len(psych_levels)}")
        
        # ====================================================================
        # 6. VALIDA E PONTUA NÍVEIS
        # ====================================================================
        
        validated_levels = self._validate_and_score_levels(
            all_levels,
            df_recent,
            current_price
        )
        
        # ====================================================================
        # 7. SEPARA SUPORTE E RESISTÊNCIA
        # ====================================================================
        
        support_levels = [
            lvl for lvl in validated_levels
            if lvl.price < current_price
        ]
        
        resistance_levels = [
            lvl for lvl in validated_levels
            if lvl.price > current_price
        ]
        
        # Ordena por distância do preço atual
        support_levels.sort(key=lambda x: x.price, reverse=True)
        resistance_levels.sort(key=lambda x: x.price)
        
        # ====================================================================
        # 8. IDENTIFICA NÍVEIS MAIS PRÓXIMOS
        # ====================================================================
        
        nearest_support = support_levels[0] if support_levels else None
        nearest_resistance = resistance_levels[0] if resistance_levels else None
        
        # Verifica se está em zona
        in_support_zone = (
            nearest_support.is_price_in_zone(current_price, self.zone_tolerance)
            if nearest_support else False
        )
        
        in_resistance_zone = (
            nearest_resistance.is_price_in_zone(current_price, self.zone_tolerance)
            if nearest_resistance else False
        )
        
        # Verifica consolidação (entre S/R próximos)
        in_consolidation = False
        if nearest_support and nearest_resistance:
            range_pct = (nearest_resistance.price - nearest_support.price) / current_price
            in_consolidation = range_pct < 0.02  # Range < 2%
        
        # ====================================================================
        # 9. CRIA RESULTADO
        # ====================================================================
        
        analysis = SRAnalysis(
            support_levels=support_levels[:10],  # Top 10 suportes
            resistance_levels=resistance_levels[:10],  # Top 10 resistências
            current_price=current_price,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            in_support_zone=in_support_zone,
            in_resistance_zone=in_resistance_zone,
            in_consolidation=in_consolidation
        )
        
        self.last_analysis = analysis
        
        # Log resumo
        logger.success(
            f"✅ S/R detectados: "
            f"{len(support_levels)} suportes, {len(resistance_levels)} resistências"
        )
        
        if nearest_support:
            logger.info(f"   Suporte mais próximo: {nearest_support.price:.2f} ({nearest_support.strength.value})")
        
        if nearest_resistance:
            logger.info(f"   Resistência mais próxima: {nearest_resistance.price:.2f} ({nearest_resistance.strength.value})")
        
        if in_consolidation:
            logger.info(f"   ⚠️ Preço em zona de consolidação")
        
        return analysis
    
    # ========================================================================
    # DETECÇÃO POR SWING HIGHS/LOWS
    # ========================================================================
    
    def _detect_swing_levels(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> List[SupportResistanceLevel]:
        """Detecta níveis baseados em swing highs/lows com clustering"""
        
        # Identifica swing highs e lows
        swing_highs = []
        swing_lows = []
        
        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            # Swing high
            is_swing_high = True
            for j in range(1, self.swing_lookback + 1):
                if (df.iloc[i]['high'] <= df.iloc[i - j]['high'] or
                    df.iloc[i]['high'] <= df.iloc[i + j]['high']):
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append({
                    'price': df.iloc[i]['high'],
                    'index': i,
                    'time': df.index[i]
                })
            
            # Swing low
            is_swing_low = True
            for j in range(1, self.swing_lookback + 1):
                if (df.iloc[i]['low'] >= df.iloc[i - j]['low'] or
                    df.iloc[i]['low'] >= df.iloc[i + j]['low']):
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append({
                    'price': df.iloc[i]['low'],
                    'index': i,
                    'time': df.index[i]
                })
        
        if not swing_highs and not swing_lows:
            return []
        
        # Aplica clustering
        levels = []
        
        if swing_highs:
            clustered_highs = self._cluster_prices([s['price'] for s in swing_highs])
            for cluster_price in clustered_highs:
                levels.append(SupportResistanceLevel(
                    price=cluster_price,
                    level_type=LevelType.RESISTANCE if cluster_price > current_price else LevelType.SUPPORT,
                    strength=LevelStrength.MEDIUM,
                    origin=LevelOrigin.SWING_POINT,
                    zone_min=cluster_price * (1 - self.zone_tolerance),
                    zone_max=cluster_price * (1 + self.zone_tolerance)
                ))
        
        if swing_lows:
            clustered_lows = self._cluster_prices([s['price'] for s in swing_lows])
            for cluster_price in clustered_lows:
                levels.append(SupportResistanceLevel(
                    price=cluster_price,
                    level_type=LevelType.SUPPORT if cluster_price < current_price else LevelType.RESISTANCE,
                    strength=LevelStrength.MEDIUM,
                    origin=LevelOrigin.SWING_POINT,
                    zone_min=cluster_price * (1 - self.zone_tolerance),
                    zone_max=cluster_price * (1 + self.zone_tolerance)
                ))
        
        return levels
    
    def _cluster_prices(self, prices: List[float]) -> List[float]:
        """Agrupa preços usando clustering"""
        if len(prices) < 2:
            return prices
        
        prices_array = np.array(prices).reshape(-1, 1)
        
        if self.clustering_method == "kmeans":
            # K-Means
            n_clusters = min(len(prices) // 3, 8)  # Max 8 clusters
            if n_clusters < 2:
                return prices
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(prices_array)
            
            return sorted(kmeans.cluster_centers_.flatten().tolist())
        
        else:
            # DBSCAN (melhor para densidade)
            scaler = StandardScaler()
            prices_scaled = scaler.fit_transform(prices_array)
            
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            clusters = dbscan.fit_predict(prices_scaled)
            
            # Calcula média de cada cluster
            unique_clusters = set(clusters)
            unique_clusters.discard(-1)  # Remove noise
            
            cluster_centers = []
            for cluster_id in unique_clusters:
                cluster_prices = [p for p, c in zip(prices, clusters) if c == cluster_id]
                cluster_centers.append(np.mean(cluster_prices))
            
            return sorted(cluster_centers) if cluster_centers else prices
    
    # ========================================================================
    # VOLUME PROFILE
    # ========================================================================
    
    def _detect_volume_levels(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> List[SupportResistanceLevel]:
        """Detecta níveis baseados em Volume Profile"""
        
        # Cria bins de preço
        price_range = df['high'].max() - df['low'].min()
        n_bins = 50
        bin_size = price_range / n_bins
        
        # Agrega volume por bin
        volume_profile = {}
        
        for _, row in df.iterrows():
            # Distribui volume do candle entre os bins que ele cobre
            low_bin = int((row['low'] - df['low'].min()) / bin_size)
            high_bin = int((row['high'] - df['low'].min()) / bin_size)
            
            for bin_idx in range(low_bin, high_bin + 1):
                if bin_idx not in volume_profile:
                    volume_profile[bin_idx] = 0
                volume_profile[bin_idx] += row['volume'] / (high_bin - low_bin + 1)
        
        # Encontra POC (Point of Control) - maior volume
        poc_bin = max(volume_profile.items(), key=lambda x: x[1])[0]
        poc_price = df['low'].min() + (poc_bin * bin_size) + (bin_size / 2)
        
        # Encontra VAH e VAL (Value Area High/Low) - 70% do volume
        sorted_bins = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(volume_profile.values())
        cumulative_volume = 0
        value_area_bins = []
        
        for bin_idx, volume in sorted_bins:
            cumulative_volume += volume
            value_area_bins.append(bin_idx)
            if cumulative_volume >= total_volume * 0.7:
                break
        
        vah_bin = max(value_area_bins)
        val_bin = min(value_area_bins)
        
        vah_price = df['low'].min() + (vah_bin * bin_size) + (bin_size / 2)
        val_price = df['low'].min() + (val_bin * bin_size) + (bin_size / 2)
        
        # Cria níveis
        levels = []
        
        # POC
        levels.append(SupportResistanceLevel(
            price=poc_price,
            level_type=LevelType.PIVOT if abs(poc_price - current_price) / current_price < 0.01 else (
                LevelType.SUPPORT if poc_price < current_price else LevelType.RESISTANCE
            ),
            strength=LevelStrength.STRONG,
            origin=LevelOrigin.VOLUME_PROFILE,
            zone_min=poc_price * (1 - self.zone_tolerance),
            zone_max=poc_price * (1 + self.zone_tolerance),
            volume_at_level=volume_profile[poc_bin]
        ))
        
        # VAH
        if vah_price != poc_price:
            levels.append(SupportResistanceLevel(
                price=vah_price,
                level_type=LevelType.RESISTANCE if vah_price > current_price else LevelType.SUPPORT,
                strength=LevelStrength.MEDIUM,
                origin=LevelOrigin.VOLUME_PROFILE,
                zone_min=vah_price * (1 - self.zone_tolerance),
                zone_max=vah_price * (1 + self.zone_tolerance),
                volume_at_level=volume_profile[vah_bin]
            ))
        
        # VAL
        if val_price != poc_price:
            levels.append(SupportResistanceLevel(
                price=val_price,
                level_type=LevelType.SUPPORT if val_price < current_price else LevelType.RESISTANCE,
                strength=LevelStrength.MEDIUM,
                origin=LevelOrigin.VOLUME_PROFILE,
                zone_min=val_price * (1 - self.zone_tolerance),
                zone_max=val_price * (1 + self.zone_tolerance),
                volume_at_level=volume_profile[val_bin]
            ))
        
        return levels
    
    # ========================================================================
    # PIVOT POINTS
    # ========================================================================
    
    def _calculate_pivot_points(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> List[SupportResistanceLevel]:
        """Calcula Pivot Points (Standard)"""
        
        # Usa dados do "dia" anterior (últimas 24h ou último período)
        period_data = df.tail(96)  # ~24h em 15min candles
        
        high = period_data['high'].max()
        low = period_data['low'].min()
        close = period_data['close'].iloc[-1]
        
        # Standard Pivot Points
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        levels = []
        
        # Pivot
        levels.append(SupportResistanceLevel(
            price=pivot,
            level_type=LevelType.PIVOT,
            strength=LevelStrength.STRONG,
            origin=LevelOrigin.PIVOT_POINT,
            zone_min=pivot * (1 - self.zone_tolerance),
            zone_max=pivot * (1 + self.zone_tolerance)
        ))
        
        # Resistências
        for i, r in enumerate([r1, r2, r3], 1):
            if r > current_price * 0.95:  # Dentro de 5% do preço atual
                levels.append(SupportResistanceLevel(
                    price=r,
                    level_type=LevelType.RESISTANCE,
                    strength=LevelStrength.MEDIUM if i <= 2 else LevelStrength.WEAK,
                    origin=LevelOrigin.PIVOT_POINT,
                    zone_min=r * (1 - self.zone_tolerance),
                    zone_max=r * (1 + self.zone_tolerance)
                ))
        
        # Suportes
        for i, s in enumerate([s1, s2, s3], 1):
            if s < current_price * 1.05:  # Dentro de 5% do preço atual
                levels.append(SupportResistanceLevel(
                    price=s,
                    level_type=LevelType.SUPPORT,
                    strength=LevelStrength.MEDIUM if i <= 2 else LevelStrength.WEAK,
                    origin=LevelOrigin.PIVOT_POINT,
                    zone_min=s * (1 - self.zone_tolerance),
                    zone_max=s * (1 + self.zone_tolerance)
                ))
        
        return levels
    
    # ========================================================================
    # FIBONACCI RETRACEMENTS
    # ========================================================================
    
    def _calculate_fibonacci_levels(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> List[SupportResistanceLevel]:
        """Calcula Fibonacci Retracements"""
        
        # Encontra swing high e low recentes
        lookback = min(100, len(df))
        recent = df.tail(lookback)
        
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        
        diff = swing_high - swing_low
        
        # Níveis de Fibonacci
        fib_levels = {
            0.236: swing_high - (diff * 0.236),
            0.382: swing_high - (diff * 0.382),
            0.500: swing_high - (diff * 0.500),
            0.618: swing_high - (diff * 0.618),
            0.786: swing_high - (diff * 0.786)
        }
        
        levels = []
        
        for ratio, price in fib_levels.items():
            # Só adiciona se estiver próximo do preço atual (dentro de 10%)
            if abs(price - current_price) / current_price < 0.10:
                level_type = LevelType.SUPPORT if price < current_price else LevelType.RESISTANCE
                
                # 0.5 e 0.618 são mais fortes
                strength = (
                    LevelStrength.STRONG if ratio in [0.5, 0.618]
                    else LevelStrength.MEDIUM
                )
                
                levels.append(SupportResistanceLevel(
                    price=price,
                    level_type=level_type,
                    strength=strength,
                    origin=LevelOrigin.FIBONACCI,
                    zone_min=price * (1 - self.zone_tolerance),
                    zone_max=price * (1 + self.zone_tolerance)
                ))
        
        return levels
    
    # ========================================================================
    # NÍVEIS PSICOLÓGICOS
    # ========================================================================
    
    def _detect_psychological_levels(
        self,
        current_price: float
    ) -> List[SupportResistanceLevel]:
        """Detecta níveis psicológicos (números redondos)"""
        
        levels = []
        
        # Determina magnitude (ex: 40000, 41000, 42000)
        if current_price > 10000:
            step = 1000
        elif current_price > 1000:
            step = 100
        elif current_price > 100:
            step = 10
        else:
            step = 1
        
        # Níveis acima e abaixo
        base = int(current_price / step) * step
        
        for i in range(-3, 4):  # 3 níveis acima e abaixo
            level_price = base + (i * step)
            
            if level_price <= 0:
                continue
            
            # Só adiciona se estiver próximo (dentro de 5%)
            if abs(level_price - current_price) / current_price > 0.05:
                continue
            
            level_type = (
                LevelType.SUPPORT if level_price < current_price
                else LevelType.RESISTANCE if level_price > current_price
                else LevelType.PIVOT
            )
            
            # Números "mais redondos" são mais fortes
            strength = LevelStrength.MEDIUM if level_price % (step * 5) == 0 else LevelStrength.WEAK
            
            levels.append(SupportResistanceLevel(
                price=float(level_price),
                level_type=level_type,
                strength=strength,
                origin=LevelOrigin.PSYCHOLOGICAL,
                zone_min=level_price * (1 - self.zone_tolerance / 2),  # Zona mais estreita
                zone_max=level_price * (1 + self.zone_tolerance / 2)
            ))
        
        return levels
    
    # ========================================================================
    # VALIDAÇÃO E PONTUAÇÃO
    # ========================================================================
    
    def _validate_and_score_levels(
        self,
        levels: List[SupportResistanceLevel],
        df: pd.DataFrame,
        current_price: float
    ) -> List[SupportResistanceLevel]:
        """Valida níveis e calcula score de relevância"""
        
        validated = []
        
        for level in levels:
            # Conta toques
            touches = self._count_touches(level, df)
            level.touches = touches
            
            # Calcula age
            level.age_candles = len(df)
            
            # Calcula score
            score = 0.0
            
            # Origem (peso)
            origin_weights = {
                LevelOrigin.VOLUME_PROFILE: 1.5,
                LevelOrigin.SWING_POINT: 1.2,
                LevelOrigin.PIVOT_POINT: 1.0,
                LevelOrigin.FIBONACCI: 0.8,
                LevelOrigin.PSYCHOLOGICAL: 0.6,
                LevelOrigin.CONSOLIDATION: 1.0
            }
            score += origin_weights.get(level.origin, 1.0)
            
            # Toques (mais toques = mais forte)
            score += min(touches * 0.3, 2.0)
            
            # Força do nível
            strength_weights = {
                LevelStrength.VERY_STRONG: 2.0,
                LevelStrength.STRONG: 1.5,
                LevelStrength.MEDIUM: 1.0,
                LevelStrength.WEAK: 0.5
            }
            score += strength_weights.get(level.strength, 1.0)
            
            # Proximidade (mais próximo = mais relevante)
            distance = level.distance_from_price(current_price)
            if distance < 0.01:  # Muito próximo (< 1%)
                score += 2.0
            elif distance < 0.02:  # Próximo (< 2%)
                score += 1.5
            elif distance < 0.05:  # Médio (< 5%)
                score += 1.0
            else:
                score += 0.5
            
            level.relevance_score = score
            
            # Valida (mínimo de toques ou score alto)
            if touches >= self.min_touches or score >= 2.0:
                validated.append(level)
        
        # Remove duplicatas (níveis muito próximos)
        validated = self._remove_duplicates(validated)
        
        # Ordena por score
        validated.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return validated
    
    def _count_touches(
        self,
        level: SupportResistanceLevel,
        df: pd.DataFrame
    ) -> int:
        """Conta quantas vezes o preço tocou o nível"""
        touches = 0
        
        for _, row in df.iterrows():
            # Verifica se high ou low estão na zona
            if (level.zone_min <= row['high'] <= level.zone_max or
                level.zone_min <= row['low'] <= level.zone_max):
                touches += 1
        
        return touches
    
    def _remove_duplicates(
        self,
        levels: List[SupportResistanceLevel]
    ) -> List[SupportResistanceLevel]:
        """Remove níveis duplicados/muito próximos"""
        if not levels:
            return []
        
        # Ordena por score
        sorted_levels = sorted(levels, key=lambda x: x.relevance_score, reverse=True)
        
        unique_levels = []
        
        for level in sorted_levels:
            # Verifica se já existe um nível próximo
            is_duplicate = False
            for existing in unique_levels:
                if abs(level.price - existing.price) / level.price < self.zone_tolerance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_levels.append(level)
        
        return unique_levels
    
    # ========================================================================
    # UTILITÁRIOS
    # ========================================================================
    
    def get_last_analysis(self) -> Optional[SRAnalysis]:
        """Retorna última análise"""
        return self.last_analysis


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Exemplo de uso do Support/Resistance Detector
    """
    from loguru import logger
    
    logger.add("logs/support_resistance.log", rotation="1 day")
    
    # Dados fictícios
    dates = pd.date_range('2024-01-01', periods=500, freq='15min')
    np.random.seed(42)
    
    # Simula preço com alguns níveis claros
    base_price = 42000
    trend = np.linspace(0, 2000, 500)
    noise = np.random.randn(500) * 50
    
    # Adiciona alguns "níveis" artificiais
    price = base_price + trend + noise
    for i in range(100, 500, 100):
        price[i-5:i+5] = 42000 + (i * 4)  # Cria consolidações
    
    df = pd.DataFrame({
        'open': price - np.random.rand(500) * 20,
        'high': price + np.random.rand(500) * 50,
        'low': price - np.random.rand(500) * 50,
        'close': price,
        'volume': 1000 + np.random.rand(500) * 500
    }, index=dates)
    
    # Cria detector
    detector = SupportResistanceDetector(
        lookback_periods=200,
        clustering_method="dbscan",
        enable_volume_profile=True,
        enable_pivot_points=True,
        enable_fibonacci=True,
        enable_psychological=True
    )
    
    # Analisa
    current_price = df.iloc[-1]['close']
    analysis = detector.analyze(df, current_price)
    
    # Mostra resultados
    logger.info("\n" + "=" * 70)
    logger.info("📊 ANÁLISE DE SUPORTE E RESISTÊNCIA")
    logger.info("=" * 70)
    logger.info(f"Preço atual: ${current_price:.2f}")
    logger.info("")
    
    if analysis.nearest_support:
        logger.info(f"🟢 Suporte mais próximo:")
        logger.info(f"   Preço: ${analysis.nearest_support.price:.2f}")
        logger.info(f"   Distância: {analysis.nearest_support.distance_from_price(current_price)*100:.2f}%")
        logger.info(f"   Força: {analysis.nearest_support.strength.value}")
        logger.info(f"   Origem: {analysis.nearest_support.origin.value}")
        logger.info(f"   Toques: {analysis.nearest_support.touches}")
    
    logger.info("")
    
    if analysis.nearest_resistance:
        logger.info(f"🔴 Resistência mais próxima:")
        logger.info(f"   Preço: ${analysis.nearest_resistance.price:.2f}")
        logger.info(f"   Distância: {analysis.nearest_resistance.distance_from_price(current_price)*100:.2f}%")
        logger.info(f"   Força: {analysis.nearest_resistance.strength.value}")
        logger.info(f"   Origem: {analysis.nearest_resistance.origin.value}")
        logger.info(f"   Toques: {analysis.nearest_resistance.touches}")
    
    logger.info("")
    logger.info(f"📍 Status:")
    logger.info(f"   Em zona de suporte: {analysis.in_support_zone}")
    logger.info(f"   Em zona de resistência: {analysis.in_resistance_zone}")
    logger.info(f"   Em consolidação: {analysis.in_consolidation}")
    
    logger.info("")
    logger.info(f"📋 Top 5 Níveis por Relevância:")
    for i, level in enumerate(analysis.get_all_levels()[:5], 1):
        symbol = "🟢" if level.level_type == LevelType.SUPPORT else "🔴" if level.level_type == LevelType.RESISTANCE else "🟡"
        logger.info(
            f"   {i}. {symbol} ${level.price:.2f} - "
            f"{level.level_type.value} ({level.strength.value}) - "
            f"Score: {level.relevance_score:.2f}"
        )