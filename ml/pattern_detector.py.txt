"""
ML Pattern Detector - Machine Learning para Detecção de Padrões
===============================================================

Módulo avançado usando Machine Learning para:
- Detectar topos e fundos com maior precisão
- Classificar padrões de candle complexos
- Prever probabilidade de reversão
- Feature engineering automático
- Treinamento e persistência de modelos

Modelos suportados:
- Random Forest
- XGBoost
- LightGBM
- Neural Network (opcional)

Autor: Trading Bot Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from loguru import logger

# Imports opcionais (instalar se disponível)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost não instalado. Use: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM não instalado. Use: pip install lightgbm")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MLPrediction:
    """Resultado de uma predição do modelo ML"""
    timestamp: int
    predicted_class: str  # "TOP", "BOTTOM", "NEUTRAL"
    probability: float  # 0.0 a 1.0
    confidence_score: float  # Score ajustado
    features: Dict[str, float]
    
    def is_top(self, threshold: float = 0.6) -> bool:
        """Retorna True se prediz topo com confiança >= threshold"""
        return self.predicted_class == "TOP" and self.probability >= threshold
    
    def is_bottom(self, threshold: float = 0.6) -> bool:
        """Retorna True se prediz fundo com confiança >= threshold"""
        return self.predicted_class == "BOTTOM" and self.probability >= threshold


# ============================================================================
# CLASSE PRINCIPAL: MLPatternDetector
# ============================================================================

class MLPatternDetector:
    """
    Detector de padrões usando Machine Learning
    """
    
    def __init__(
        self,
        model_type: Literal["random_forest", "xgboost", "lightgbm", "gradient_boosting"] = "random_forest",
        lookback_periods: int = 20,
        model_path: Optional[str] = None,
        auto_train: bool = False
    ):
        """
        Inicializa o detector ML
        
        Args:
            model_type: Tipo de modelo a usar
            lookback_periods: Períodos para criar features
            model_path: Caminho para carregar modelo pré-treinado
            auto_train: Se True, treina automaticamente com dados
        """
        self.model_type = model_type
        self.lookback_periods = lookback_periods
        self.model_path = model_path
        self.auto_train = auto_train
        
        # Modelo e scaler
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Feature names (para referência)
        self.feature_names: List[str] = []
        
        # Classes
        self.classes = ["NEUTRAL", "TOP", "BOTTOM"]
        
        # Tenta carregar modelo existente
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        logger.info(
            f"MLPatternDetector inicializado: "
            f"model={model_type}, lookback={lookback_periods}, "
            f"trained={self.is_trained}"
        )
    
    # ========================================================================
    # FEATURE ENGINEERING
    # ========================================================================
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features para machine learning
        
        Args:
            df: DataFrame com OHLCV e indicadores
        
        Returns:
            DataFrame com features
        """
        df = df.copy()
        
        # ====================================================================
        # 1. FEATURES DE PREÇO
        # ====================================================================
        
        # Retornos
        df['returns'] = df['close'].pct_change()
        df['returns_abs'] = df['returns'].abs()
        
        # Log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_ma'] = df['hl_range'].rolling(window=10).mean()
        
        # Body do candle
        df['body'] = abs(df['close'] - df['open']) / df['close']
        df['body_ma'] = df['body'].rolling(window=10).mean()
        
        # Upper/Lower wicks
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Tendência do candle
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        
        # ====================================================================
        # 2. FEATURES DE MOMENTUM
        # ====================================================================
        
        # RSI (já calculado, mas adiciona variações)
        if 'rsi' in df.columns:
            df['rsi_ma'] = df['rsi'].rolling(window=5).mean()
            df['rsi_std'] = df['rsi'].rolling(window=10).std()
            df['rsi_change'] = df['rsi'].diff()
        
        # MACD
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_diff_norm'] = df['macd_diff'] / df['close']
            df['macd_hist_ma'] = df['macd_diff'].rolling(window=5).mean()
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period)
        
        # ====================================================================
        # 3. FEATURES DE VOLATILIDADE
        # ====================================================================
        
        # ATR normalizado
        if 'atr' in df.columns:
            df['atr_norm'] = df['atr'] / df['close']
            df['atr_ma'] = df['atr'].rolling(window=10).mean()
            df['atr_std'] = df['atr'].rolling(window=10).std()
        
        # Volatilidade realizada
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Bollinger Band width
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ====================================================================
        # 4. FEATURES DE VOLUME
        # ====================================================================
        
        # Volume normalizado
        df['volume_norm'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['volume_change'] = df['volume'].pct_change()
        
        # Volume weighted price
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['price_to_vwap'] = df['close'] / df['vwap']
        
        # ====================================================================
        # 5. FEATURES DE PADRÃO
        # ====================================================================
        
        # Sequências de candles
        df['consecutive_up'] = (df['close'] > df['open']).astype(int).groupby(
            (df['close'] <= df['open']).cumsum()
        ).cumsum()
        
        df['consecutive_down'] = (df['close'] < df['open']).astype(int).groupby(
            (df['close'] >= df['open']).cumsum()
        ).cumsum()
        
        # Distância de EMAs
        if 'ema_fast' in df.columns and 'ema_medium' in df.columns:
            df['ema_distance'] = (df['ema_fast'] - df['ema_medium']) / df['close']
            df['price_to_ema_fast'] = (df['close'] - df['ema_fast']) / df['close']
            df['price_to_ema_medium'] = (df['close'] - df['ema_medium']) / df['close']
        
        # ====================================================================
        # 6. FEATURES LAG (valores passados)
        # ====================================================================
        
        lag_features = ['returns', 'volume_norm', 'rsi', 'body']
        for feature in lag_features:
            if feature in df.columns:
                for lag in [1, 2, 3, 5]:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        # ====================================================================
        # 7. FEATURES ESTATÍSTICAS (rolling)
        # ====================================================================
        
        rolling_features = ['close', 'volume', 'returns']
        windows = [5, 10, 20]
        
        for feature in rolling_features:
            if feature in df.columns:
                for window in windows:
                    df[f'{feature}_ma_{window}'] = df[feature].rolling(window=window).mean()
                    df[f'{feature}_std_{window}'] = df[feature].rolling(window=window).std()
                    df[f'{feature}_min_{window}'] = df[feature].rolling(window=window).min()
                    df[f'{feature}_max_{window}'] = df[feature].rolling(window=window).max()
        
        # ====================================================================
        # 8. FEATURES DE POSIÇÃO RELATIVA
        # ====================================================================
        
        # Posição em relação a máximas/mínimas recentes
        for window in [10, 20, 50]:
            df[f'high_{window}'] = df['high'].rolling(window=window).max()
            df[f'low_{window}'] = df['low'].rolling(window=window).min()
            df[f'position_in_range_{window}'] = (
                (df['close'] - df[f'low_{window}']) / 
                (df[f'high_{window}'] - df[f'low_{window}'])
            )
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara features finais para o modelo
        
        Args:
            df: DataFrame com features criadas
        
        Returns:
            DataFrame apenas com features válidas
        """
        # Remove colunas OHLCV originais
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        
        # Seleciona apenas colunas numéricas
        feature_cols = [
            col for col in df.columns 
            if col not in exclude_cols and df[col].dtype in ['float64', 'int64']
        ]
        
        df_features = df[feature_cols].copy()
        
        # Remove linhas com NaN
        df_features = df_features.dropna()
        
        # Armazena nomes das features
        self.feature_names = df_features.columns.tolist()
        
        logger.info(f"Features preparadas: {len(self.feature_names)} features")
        
        return df_features
    
    # ========================================================================
    # LABELING (criar labels para treinamento)
    # ========================================================================
    
    def create_labels(
        self,
        df: pd.DataFrame,
        future_periods: int = 5,
        threshold_pct: float = 0.5
    ) -> pd.Series:
        """
        Cria labels para treinamento supervisionado
        
        Labels:
        - TOP: Se preço cair > threshold_pct nos próximos N períodos
        - BOTTOM: Se preço subir > threshold_pct nos próximos N períodos
        - NEUTRAL: Caso contrário
        
        Args:
            df: DataFrame com dados
            future_periods: Períodos futuros a considerar
            threshold_pct: Threshold em % para classificar
        
        Returns:
            Series com labels
        """
        labels = []
        
        for i in range(len(df)):
            # Se não há dados futuros suficientes
            if i + future_periods >= len(df):
                labels.append("NEUTRAL")
                continue
            
            current_price = df.iloc[i]['close']
            
            # Preços futuros
            future_prices = df.iloc[i+1:i+future_periods+1]['close']
            max_future = future_prices.max()
            min_future = future_prices.min()
            
            # Variação percentual
            up_pct = ((max_future - current_price) / current_price) * 100
            down_pct = ((current_price - min_future) / current_price) * 100
            
            # Classifica
            if down_pct > threshold_pct and down_pct > up_pct:
                labels.append("TOP")
            elif up_pct > threshold_pct and up_pct > down_pct:
                labels.append("BOTTOM")
            else:
                labels.append("NEUTRAL")
        
        return pd.Series(labels, index=df.index)
    
    # ========================================================================
    # TREINAMENTO
    # ========================================================================
    
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        future_periods: int = 5,
        threshold_pct: float = 0.5
    ) -> Dict:
        """
        Treina o modelo com dados históricos
        
        Args:
            df: DataFrame com OHLCV
            test_size: Proporção para test set
            future_periods: Períodos futuros para criar labels
            threshold_pct: Threshold para classificação
        
        Returns:
            Dict com métricas de performance
        """
        logger.info("🚀 Iniciando treinamento do modelo ML...")
        
        # Cria features
        logger.info("Criando features...")
        df_features = self.create_features(df)
        
        # Cria labels
        logger.info("Criando labels...")
        labels = self.create_labels(df, future_periods, threshold_pct)
        
        # Prepara features
        X = self.prepare_features(df_features)
        
        # Alinha labels com features (remove NaN)
        y = labels.loc[X.index]
        
        logger.info(f"Dataset preparado: {len(X)} amostras, {len(X.columns)} features")
        logger.info(f"Distribuição de classes:\n{y.value_counts()}")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Normaliza features
        logger.info("Normalizando features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Cria modelo
        logger.info(f"Criando modelo: {self.model_type}")
        self.model = self._create_model()
        
        # Treina
        logger.info("Treinando modelo...")
        self.model.fit(X_train_scaled, y_train)
        
        # Avalia
        logger.info("Avaliando modelo...")
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importance = self._get_feature_importance()
        
        self.is_trained = True
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance,
            'n_samples': len(X),
            'n_features': len(X.columns),
            'class_distribution': y.value_counts().to_dict()
        }
        
        logger.success(f"✅ Modelo treinado com sucesso!")
        logger.info(f"   Train Accuracy: {train_score:.3f}")
        logger.info(f"   Test Accuracy: {test_score:.3f}")
        
        return metrics
    
    def _create_model(self):
        """Cria o modelo baseado no tipo especificado"""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                random_state=42
            )
        
        elif self.model_type == "xgboost" and HAS_XGB:
            return xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        
        elif self.model_type == "lightgbm" and HAS_LGB:
            return lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        else:
            logger.warning(f"Modelo {self.model_type} não disponível, usando Random Forest")
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Retorna importância das features"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_imp = dict(zip(self.feature_names, importances))
            # Ordena por importância
            feature_imp = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True))
            return feature_imp
        return {}
    
    # ========================================================================
    # PREDIÇÃO
    # ========================================================================
    
    def predict(
        self,
        df: pd.DataFrame,
        return_proba: bool = True
    ) -> Optional[MLPrediction]:
        """
        Faz predição para o último candle
        
        Args:
            df: DataFrame com OHLCV e indicadores
            return_proba: Se True, retorna probabilidades
        
        Returns:
            MLPrediction ou None
        """
        if not self.is_trained:
            logger.warning("Modelo não treinado. Use train() primeiro.")
            return None
        
        # Cria features
        df_features = self.create_features(df)
        X = self.prepare_features(df_features)
        
        if X.empty:
            logger.warning("Não foi possível criar features válidas")
            return None
        
        # Pega última linha
        X_last = X.iloc[[-1]]
        
        # Normaliza
        X_scaled = self.scaler.transform(X_last)
        
        # Prediz
        pred_class = self.model.predict(X_scaled)[0]
        pred_proba = self.model.predict_proba(X_scaled)[0]
        
        # Índice da classe predita
        class_idx = self.model.classes_.tolist().index(pred_class)
        probability = pred_proba[class_idx]
        
        # Confidence score (diferença entre top 2 probabilidades)
        sorted_proba = sorted(pred_proba, reverse=True)
        confidence_score = sorted_proba[0] - sorted_proba[1]
        
        # Features usadas
        features_dict = X_last.iloc[0].to_dict()
        
        prediction = MLPrediction(
            timestamp=int(df.index[-1].timestamp() * 1000) if hasattr(df.index[-1], 'timestamp') else 0,
            predicted_class=pred_class,
            probability=probability,
            confidence_score=confidence_score,
            features=features_dict
        )
        
        return prediction
    
    # ========================================================================
    # PERSISTÊNCIA
    # ========================================================================
    
    def save_model(self, path: str):
        """Salva modelo treinado"""
        if not self.is_trained:
            logger.warning("Modelo não treinado. Nada para salvar.")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'lookback_periods': self.lookback_periods
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, path)
        
        logger.success(f"✅ Modelo salvo: {path}")
    
    def load_model(self, path: str):
        """Carrega modelo treinado"""
        if not Path(path).exists():
            logger.error(f"Modelo não encontrado: {path}")
            return
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data.get('model_type', self.model_type)
        self.lookback_periods = model_data.get('lookback_periods', self.lookback_periods)
        self.is_trained = True
        
        logger.success(f"✅ Modelo carregado: {path}")


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Exemplo de treinamento e uso do ML Pattern Detector
    """
    from loguru import logger
    
    logger.add("logs/ml_pattern_detector.log", rotation="1 day")
    
    # Dados fictícios (substituir por dados reais)
    dates = pd.date_range('2024-01-01', periods=5000, freq='15min')
    np.random.seed(42)
    
    close_prices = 42000 + np.cumsum(np.random.randn(5000) * 50)
    
    df = pd.DataFrame({
        'open': close_prices - np.random.rand(5000) * 20,
        'high': close_prices + np.random.rand(5000) * 50,
        'low': close_prices - np.random.rand(5000) * 50,
        'close': close_prices,
        'volume': np.random.rand(5000) * 1000
    }, index=dates)
    
    # Calcula indicadores básicos (usar strategy.calculate_indicators na prática)
    df['rsi'] = 50 + np.random.randn(5000) * 15
    df['macd'] = np.random.randn(5000) * 10
    df['macd_signal'] = df['macd'].rolling(9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['atr'] = np.random.rand(5000) * 100
    
    # Cria detector
    detector = MLPatternDetector(model_type="random_forest")
    
    # Treina
    metrics = detector.train(df, test_size=0.2)
    
    logger.info(f"Métricas de treinamento: {metrics}")
    
    # Salva modelo
    detector.save_model("models/ml_pattern_detector.pkl")
    
    # Faz predição
    prediction = detector.predict(df)
    
    if prediction:
        logger.info(f"Predição: {prediction.predicted_class} (prob: {prediction.probability:.2%})")