"""
Grid Optimizer - Auto-Otimização de Parâmetros via Grid Search
==============================================================

Módulo para otimização automática de parâmetros da estratégia:
- Grid Search (busca exaustiva)
- Random Search (busca aleatória)
- Bayesian Optimization (busca inteligente)
- Walk-Forward Analysis
- Parallel processing
- Validação cruzada temporal

Otimiza para maximizar:
- Sharpe Ratio
- Profit Factor
- Win Rate
- Total Return
- Custom metrics

Autor: Trading Bot Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

# Imports do projeto
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.backtester import Backtester, BacktestConfig, BacktestResults
from strategy.strategy_core import ScalpingStrategy


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ParameterSpace:
    """Define espaço de parâmetros para otimização"""
    name: str
    values: List[Any]
    
    def __len__(self):
        return len(self.values)


@dataclass
class OptimizationResult:
    """Resultado de uma otimização"""
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    backtest_results: Optional[BacktestResults] = None
    
    def get_score(self, objective: str = "sharpe_ratio") -> float:
        """Retorna score do objetivo"""
        return self.metrics.get(objective, 0.0)
    
    def to_dict(self) -> Dict:
        return {
            'parameters': self.parameters,
            'metrics': self.metrics
        }


@dataclass
class OptimizationSummary:
    """Resumo da otimização"""
    best_parameters: Dict[str, Any]
    best_metrics: Dict[str, float]
    all_results: List[OptimizationResult] = field(default_factory=list)
    optimization_time_seconds: float = 0.0
    total_combinations: int = 0
    objective: str = "sharpe_ratio"
    
    def to_dict(self) -> Dict:
        return {
            'best_parameters': self.best_parameters,
            'best_metrics': self.best_metrics,
            'optimization_time_seconds': self.optimization_time_seconds,
            'total_combinations': self.total_combinations,
            'objective': self.objective,
            'top_10_results': [r.to_dict() for r in self.all_results[:10]]
        }
    
    def print_summary(self):
        """Imprime resumo da otimização"""
        logger.info("=" * 70)
        logger.info("🎯 RESULTADOS DA OTIMIZAÇÃO")
        logger.info("=" * 70)
        logger.info(f"Objetivo: {self.objective}")
        logger.info(f"Combinações testadas: {self.total_combinations}")
        logger.info(f"Tempo total: {self.optimization_time_seconds:.1f}s")
        logger.info("")
        logger.info("🏆 Melhores Parâmetros:")
        for param, value in self.best_parameters.items():
            logger.info(f"   {param}: {value}")
        logger.info("")
        logger.info("📊 Métricas do Melhor:")
        for metric, value in self.best_metrics.items():
            logger.info(f"   {metric}: {value:.4f}")
        logger.info("")
        logger.info("🥇 Top 5 Combinações:")
        for i, result in enumerate(self.all_results[:5], 1):
            score = result.get_score(self.objective)
            logger.info(f"   {i}. Score: {score:.4f} | Params: {result.parameters}")
        logger.info("=" * 70)


# ============================================================================
# CLASSE PRINCIPAL: GridOptimizer
# ============================================================================

class GridOptimizer:
    """
    Otimizador de parâmetros via Grid Search
    """
    
    def __init__(
        self,
        base_config: BacktestConfig,
        objective: str = "sharpe_ratio",
        n_jobs: int = -1,
        verbose: bool = True
    ):
        """
        Inicializa o otimizador
        
        Args:
            base_config: Configuração base do backtest
            objective: Métrica a otimizar (sharpe_ratio, profit_factor, win_rate, etc)
            n_jobs: Número de processos paralelos (-1 = todos os cores)
            verbose: Se True, mostra progresso
        """
        self.base_config = base_config
        self.objective = objective
        self.n_jobs = n_jobs if n_jobs != -1 else None
        self.verbose = verbose
        
        # Resultados
        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
        
        logger.info(
            f"GridOptimizer inicializado: "
            f"objetivo={objective}, n_jobs={n_jobs}"
        )
    
    # ========================================================================
    # GRID SEARCH
    # ========================================================================
    
    def grid_search(
        self,
        parameter_space: Dict[str, List[Any]],
        df_primary: pd.DataFrame,
        df_confirmation: pd.DataFrame,
        max_combinations: Optional[int] = None
    ) -> OptimizationSummary:
        """
        Grid Search: testa todas as combinações de parâmetros
        
        Args:
            parameter_space: Dict com {param_name: [values]}
            df_primary: DataFrame do timeframe primário
            df_confirmation: DataFrame do timeframe de confirmação
            max_combinations: Limite de combinações (None = todas)
        
        Returns:
            OptimizationSummary com resultados
        """
        logger.info("🚀 Iniciando Grid Search...")
        
        start_time = datetime.now()
        
        # Cria todas as combinações
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        
        all_combinations = list(product(*param_values))
        
        # Limita combinações se necessário
        if max_combinations and len(all_combinations) > max_combinations:
            logger.warning(
                f"Limitando de {len(all_combinations)} para "
                f"{max_combinations} combinações"
            )
            np.random.shuffle(all_combinations)
            all_combinations = all_combinations[:max_combinations]
        
        total_combinations = len(all_combinations)
        logger.info(f"Total de combinações: {total_combinations}")
        
        # Prepara tasks
        tasks = []
        for combination in all_combinations:
            params = dict(zip(param_names, combination))
            tasks.append((params, df_primary, df_confirmation))
        
        # Executa em paralelo
        logger.info(f"Executando backtests em paralelo (workers={self.n_jobs})...")
        
        self.results = []
        
        if self.n_jobs == 1:
            # Sequential
            for i, task in enumerate(tasks, 1):
                if self.verbose and i % 10 == 0:
                    logger.info(f"Progresso: {i}/{total_combinations}")
                result = self._run_backtest_task(task)
                if result:
                    self.results.append(result)
        else:
            # Parallel
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(self._run_backtest_task, task)
                    for task in tasks
                ]
                
                for i, future in enumerate(as_completed(futures), 1):
                    if self.verbose and i % 10 == 0:
                        logger.info(f"Progresso: {i}/{total_combinations}")
                    
                    try:
                        result = future.result()
                        if result:
                            self.results.append(result)
                    except Exception as e:
                        logger.error(f"Erro em backtest: {e}")
        
        # Ordena por objetivo
        self.results.sort(
            key=lambda r: r.get_score(self.objective),
            reverse=True
        )
        
        self.best_result = self.results[0] if self.results else None
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.success(f"✅ Grid Search concluído em {elapsed:.1f}s")
        
        # Cria resumo
        summary = OptimizationSummary(
            best_parameters=self.best_result.parameters if self.best_result else {},
            best_metrics=self.best_result.metrics if self.best_result else {},
            all_results=self.results,
            optimization_time_seconds=elapsed,
            total_combinations=total_combinations,
            objective=self.objective
        )
        
        summary.print_summary()
        
        return summary
    
    # ========================================================================
    # RANDOM SEARCH
    # ========================================================================
    
    def random_search(
        self,
        parameter_space: Dict[str, List[Any]],
        df_primary: pd.DataFrame,
        df_confirmation: pd.DataFrame,
        n_iterations: int = 100,
        seed: Optional[int] = None
    ) -> OptimizationSummary:
        """
        Random Search: testa combinações aleatórias
        
        Mais eficiente que Grid Search quando há muitos parâmetros
        
        Args:
            parameter_space: Dict com {param_name: [values]}
            df_primary: DataFrame do timeframe primário
            df_confirmation: DataFrame do timeframe de confirmação
            n_iterations: Número de combinações aleatórias a testar
            seed: Seed para reprodutibilidade
        
        Returns:
            OptimizationSummary com resultados
        """
        logger.info(f"🎲 Iniciando Random Search ({n_iterations} iterações)...")
        
        if seed is not None:
            np.random.seed(seed)
        
        start_time = datetime.now()
        
        # Gera combinações aleatórias
        param_names = list(parameter_space.keys())
        random_combinations = []
        
        for _ in range(n_iterations):
            combination = {
                name: np.random.choice(parameter_space[name])
                for name in param_names
            }
            random_combinations.append(combination)
        
        # Prepara tasks
        tasks = [
            (params, df_primary, df_confirmation)
            for params in random_combinations
        ]
        
        # Executa em paralelo
        logger.info(f"Executando backtests em paralelo...")
        
        self.results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(self._run_backtest_task, task)
                for task in tasks
            ]
            
            for i, future in enumerate(as_completed(futures), 1):
                if self.verbose and i % 10 == 0:
                    logger.info(f"Progresso: {i}/{n_iterations}")
                
                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                except Exception as e:
                    logger.error(f"Erro em backtest: {e}")
        
        # Ordena por objetivo
        self.results.sort(
            key=lambda r: r.get_score(self.objective),
            reverse=True
        )
        
        self.best_result = self.results[0] if self.results else None
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.success(f"✅ Random Search concluído em {elapsed:.1f}s")
        
        # Cria resumo
        summary = OptimizationSummary(
            best_parameters=self.best_result.parameters if self.best_result else {},
            best_metrics=self.best_result.metrics if self.best_result else {},
            all_results=self.results,
            optimization_time_seconds=elapsed,
            total_combinations=n_iterations,
            objective=self.objective
        )
        
        summary.print_summary()
        
        return summary
    
    # ========================================================================
    # WALK-FORWARD ANALYSIS
    # ========================================================================
    
    def walk_forward_analysis(
        self,
        parameter_space: Dict[str, List[Any]],
        df_primary: pd.DataFrame,
        df_confirmation: pd.DataFrame,
        train_periods: int = 3,
        test_periods: int = 1,
        max_combinations_per_window: int = 50
    ) -> Dict:
        """
        Walk-Forward Analysis: otimiza e testa em janelas móveis
        
        Evita overfitting ao testar em dados "futuros" não vistos
        
        Args:
            parameter_space: Dict com {param_name: [values]}
            df_primary: DataFrame do timeframe primário
            df_confirmation: DataFrame do timeframe de confirmação
            train_periods: Períodos para treino (ex: 3 meses)
            test_periods: Períodos para teste (ex: 1 mês)
            max_combinations_per_window: Max combinações por janela
        
        Returns:
            Dict com resultados de cada janela
        """
        logger.info("🔄 Iniciando Walk-Forward Analysis...")
        
        # Divide dados em janelas
        total_periods = len(df_primary) // 30  # Aproximadamente meses
        windows = []
        
        for i in range(0, total_periods - train_periods - test_periods + 1, test_periods):
            train_start = i * 30
            train_end = (i + train_periods) * 30
            test_start = train_end
            test_end = (i + train_periods + test_periods) * 30
            
            if test_end > len(df_primary):
                break
            
            windows.append({
                'train': (train_start, train_end),
                'test': (test_start, test_end)
            })
        
        logger.info(f"Criadas {len(windows)} janelas walk-forward")
        
        results_by_window = []
        
        for i, window in enumerate(windows, 1):
            logger.info(f"\n📊 Janela {i}/{len(windows)}")
            
            # Dados de treino
            df_train = df_primary.iloc[window['train'][0]:window['train'][1]]
            df_train_conf = df_confirmation.iloc[window['train'][0]:window['train'][1]]
            
            # Dados de teste
            df_test = df_primary.iloc[window['test'][0]:window['test'][1]]
            df_test_conf = df_confirmation.iloc[window['test'][0]:window['test'][1]]
            
            # Otimiza na janela de treino
            logger.info("   Otimizando em período de treino...")
            train_summary = self.random_search(
                parameter_space=parameter_space,
                df_primary=df_train,
                df_confirmation=df_train_conf,
                n_iterations=max_combinations_per_window
            )
            
            # Testa com melhores parâmetros no período de teste
            logger.info("   Testando em período out-of-sample...")
            best_params = train_summary.best_parameters
            
            # Cria config com melhores parâmetros
            test_config = self._create_config_with_params(best_params)
            
            # Roda backtest no período de teste
            backtester = Backtester(test_config)
            test_results = backtester.run(df_test, df_test_conf)
            
            window_result = {
                'window_id': i,
                'train_period': window['train'],
                'test_period': window['test'],
                'best_params': best_params,
                'train_metrics': train_summary.best_metrics,
                'test_metrics': {
                    'sharpe_ratio': test_results.sharpe_ratio,
                    'win_rate': test_results.win_rate,
                    'profit_factor': test_results.profit_factor,
                    'total_pnl_pct': test_results.total_pnl_pct,
                    'max_drawdown_pct': test_results.max_drawdown_pct
                }
            }
            
            results_by_window.append(window_result)
            
            logger.info(f"   Train Sharpe: {train_summary.best_metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"   Test Sharpe: {test_results.sharpe_ratio:.2f}")
        
        # Calcula métricas agregadas
        avg_test_sharpe = np.mean([w['test_metrics']['sharpe_ratio'] for w in results_by_window])
        avg_test_winrate = np.mean([w['test_metrics']['win_rate'] for w in results_by_window])
        
        logger.success(f"\n✅ Walk-Forward concluído!")
        logger.info(f"   Avg Test Sharpe: {avg_test_sharpe:.2f}")
        logger.info(f"   Avg Test Win Rate: {avg_test_winrate:.1f}%")
        
        return {
            'windows': results_by_window,
            'aggregate_metrics': {
                'avg_test_sharpe': avg_test_sharpe,
                'avg_test_winrate': avg_test_winrate
            }
        }
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def _run_backtest_task(
        self,
        task: Tuple[Dict, pd.DataFrame, pd.DataFrame]
    ) -> Optional[OptimizationResult]:
        """Executa um backtest com parâmetros específicos"""
        params, df_primary, df_confirmation = task
        
        try:
            # Cria config com parâmetros
            config = self._create_config_with_params(params)
            
            # Roda backtest
            backtester = Backtester(config)
            results = backtester.run(df_primary, df_confirmation)
            
            # Extrai métricas
            metrics = {
                'sharpe_ratio': results.sharpe_ratio,
                'sortino_ratio': results.sortino_ratio,
                'profit_factor': results.profit_factor,
                'win_rate': results.win_rate,
                'total_pnl_pct': results.total_pnl_pct,
                'max_drawdown_pct': results.max_drawdown_pct,
                'total_trades': results.total_trades,
                'avg_win': results.avg_win,
                'avg_loss': results.avg_loss
            }
            
            return OptimizationResult(
                parameters=params,
                metrics=metrics,
                backtest_results=results
            )
        
        except Exception as e:
            logger.error(f"Erro no backtest: {e}")
            return None
    
    def _create_config_with_params(self, params: Dict) -> BacktestConfig:
        """Cria BacktestConfig com parâmetros personalizados"""
        config = BacktestConfig(
            symbol=self.base_config.symbol,
            start_date=self.base_config.start_date,
            end_date=self.base_config.end_date,
            initial_capital=self.base_config.initial_capital,
            risk_per_trade=params.get('risk_per_trade', self.base_config.risk_per_trade),
            leverage=params.get('leverage', self.base_config.leverage),
            trading_fee=self.base_config.trading_fee,
            slippage_pct=self.base_config.slippage_pct,
            max_positions=self.base_config.max_positions,
            primary_timeframe=self.base_config.primary_timeframe,
            confirmation_timeframe=self.base_config.confirmation_timeframe,
            atr_multiplier_sl=params.get('atr_multiplier_sl', self.base_config.atr_multiplier_sl),
            atr_multiplier_tp1=params.get('atr_multiplier_tp1', self.base_config.atr_multiplier_tp1),
            atr_multiplier_tp2=params.get('atr_multiplier_tp2', self.base_config.atr_multiplier_tp2),
            confidence_threshold=params.get('confidence_threshold', self.base_config.confidence_threshold)
        )
        
        return config
    
    # ========================================================================
    # EXPORTAÇÃO
    # ========================================================================
    
    def export_results(
        self,
        summary: OptimizationSummary,
        output_dir: str = "optimization_results"
    ):
        """Exporta resultados da otimização"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol = self.base_config.symbol
        
        # JSON completo
        json_file = f"{output_dir}/optimization_{symbol}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info(f"📁 JSON exportado: {json_file}")
        
        # CSV com todos os resultados
        if summary.all_results:
            results_data = []
            for result in summary.all_results:
                row = {**result.parameters, **result.metrics}
                results_data.append(row)
            
            df_results = pd.DataFrame(results_data)
            csv_file = f"{output_dir}/optimization_results_{symbol}_{timestamp}.csv"
            df_results.to_csv(csv_file, index=False)
            logger.info(f"📁 CSV exportado: {csv_file}")


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def create_default_parameter_space() -> Dict[str, List[Any]]:
    """Cria espaço de parâmetros padrão para otimização"""
    return {
        'atr_multiplier_sl': [0.8, 1.0, 1.2, 1.5],
        'atr_multiplier_tp1': [0.8, 1.0, 1.2, 1.5],
        'atr_multiplier_tp2': [1.5, 2.0, 2.5, 3.0],
        'confidence_threshold': [0.5, 0.6, 0.65, 0.7],
        'risk_per_trade': [0.005, 0.01, 0.015, 0.02],
        'leverage': [3, 5, 7, 10]
    }


def create_aggressive_parameter_space() -> Dict[str, List[Any]]:
    """Espaço de parâmetros mais agressivo"""
    return {
        'atr_multiplier_sl': [0.5, 0.8, 1.0],
        'atr_multiplier_tp1': [0.5, 0.8, 1.0],
        'atr_multiplier_tp2': [1.0, 1.5, 2.0],
        'confidence_threshold': [0.5, 0.55, 0.6],
        'risk_per_trade': [0.02, 0.025, 0.03],
        'leverage': [5, 7, 10, 15]
    }


def create_conservative_parameter_space() -> Dict[str, List[Any]]:
    """Espaço de parâmetros mais conservador"""
    return {
        'atr_multiplier_sl': [1.0, 1.5, 2.0],
        'atr_multiplier_tp1': [1.0, 1.5, 2.0],
        'atr_multiplier_tp2': [2.0, 2.5, 3.0, 4.0],
        'confidence_threshold': [0.65, 0.7, 0.75, 0.8],
        'risk_per_trade': [0.005, 0.0075, 0.01],
        'leverage': [2, 3, 5]
    }


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Exemplo de otimização de parâmetros
    """
    from loguru import logger
    
    logger.add("logs/grid_optimizer.log", rotation="1 day")
    
    # Dados fictícios (substituir por dados reais)
    dates_15m = pd.date_range('2024-01-01', '2024-06-30', freq='15min')
    dates_1h = pd.date_range('2024-01-01', '2024-06-30', freq='1H')
    
    np.random.seed(42)
    close_15m = 42000 + np.cumsum(np.random.randn(len(dates_15m)) * 50)
    close_1h = 42000 + np.cumsum(np.random.randn(len(dates_1h)) * 100)
    
    df_15m = pd.DataFrame({
        'open': close_15m - np.random.rand(len(dates_15m)) * 20,
        'high': close_15m + np.random.rand(len(dates_15m)) * 50,
        'low': close_15m - np.random.rand(len(dates_15m)) * 50,
        'close': close_15m,
        'volume': np.random.rand(len(dates_15m)) * 1000
    }, index=dates_15m)
    
    df_1h = pd.DataFrame({
        'open': close_1h - np.random.rand(len(dates_1h)) * 40,
        'high': close_1h + np.random.rand(len(dates_1h)) * 100,
        'low': close_1h - np.random.rand(len(dates_1h)) * 100,
        'close': close_1h,
        'volume': np.random.rand(len(dates_1h)) * 5000
    }, index=dates_1h)
    
    # Configuração base
    base_config = BacktestConfig(
        symbol="BTCUSDT",
        start_date="2024-01-01",
        end_date="2024-06-30",
        initial_capital=10000.0
    )
    
    # Cria otimizador
    optimizer = GridOptimizer(
        base_config=base_config,
        objective="sharpe_ratio",
        n_jobs=-1
    )
    
    # Define espaço de parâmetros
    param_space = create_default_parameter_space()
    
    logger.info(f"Espaço de parâmetros: {param_space}")
    
    # Opção 1: Grid Search (testa todas as combinações)
    # summary = optimizer.grid_search(
    #     parameter_space=param_space,
    #     df_primary=df_15m,
    #     df_confirmation=df_1h,
    #     max_combinations=100  # Limita para exemplo
    # )
    
    # Opção 2: Random Search (mais rápido)
    summary = optimizer.random_search(
        parameter_space=param_space,
        df_primary=df_15m,
        df_confirmation=df_1h,
        n_iterations=50
    )
    
    # Exporta resultados
    optimizer.export_results(summary)
    
    logger.success("🎉 Otimização concluída!")