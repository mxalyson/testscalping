"""
Backtester - Simulação Histórica da Estratégia
==============================================

Módulo para backtest da estratégia com:
- Simulação de trades com dados históricos
- Cálculo de métricas de performance
- Consideração de fees e slippage
- Relatórios detalhados
- Exportação de resultados

Autor: Trading Bot Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
import json
from pathlib import Path
import sys

# Adiciona diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from strategy.strategy_core import ScalpingStrategy, SignalType, TradingSignal
from risk.risk_manager import RiskManager, TradeRecord


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BacktestConfig:
    """Configurações do backtest"""
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.01
    leverage: float = 5.0
    trading_fee: float = 0.0006  # 0.06%
    slippage_pct: float = 0.001  # 0.1%
    max_positions: int = 1  # Para backtest simples, 1 posição por vez
    primary_timeframe: str = "15"
    confirmation_timeframe: str = "60"
    
    # Strategy parameters
    atr_multiplier_sl: float = 1.0
    atr_multiplier_tp1: float = 1.0
    atr_multiplier_tp2: float = 2.0
    confidence_threshold: float = 0.6
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'risk_per_trade': self.risk_per_trade,
            'leverage': self.leverage,
            'trading_fee': self.trading_fee,
            'slippage_pct': self.slippage_pct,
            'max_positions': self.max_positions,
            'primary_timeframe': self.primary_timeframe,
            'confirmation_timeframe': self.confirmation_timeframe,
            'atr_multiplier_sl': self.atr_multiplier_sl,
            'atr_multiplier_tp1': self.atr_multiplier_tp1,
            'atr_multiplier_tp2': self.atr_multiplier_tp2,
            'confidence_threshold': self.confidence_threshold
        }


@dataclass
class BacktestTrade:
    """Representa um trade no backtest"""
    trade_id: int
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    exit_reason: str  # "TP1", "TP2", "SL", "End of data"
    pnl: float
    pnl_pct: float
    fees: float
    confidence: float
    signal_reason: str
    
    def to_dict(self) -> Dict:
        return {
            'trade_id': self.trade_id,
            'entry_time': self.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            'exit_time': self.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'leverage': self.leverage,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'fees': self.fees,
            'confidence': self.confidence,
            'signal_reason': self.signal_reason
        }


@dataclass
class BacktestResults:
    """Resultados do backtest"""
    config: BacktestConfig
    trades: List[BacktestTrade] = field(default_factory=list)
    
    # Performance metrics
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    total_fees: float = 0.0
    
    # Time metrics
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_days: int = 0
    
    def calculate_metrics(self):
        """Calcula métricas de performance"""
        if not self.trades:
            logger.warning("Nenhum trade para calcular métricas")
            return
        
        # Basic metrics
        self.total_trades = len(self.trades)
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        
        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # PnL metrics
        self.total_pnl = sum(t.pnl for t in self.trades)
        self.total_pnl_pct = (self.total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0
        self.final_capital = self.initial_capital + self.total_pnl
        
        # Win/Loss averages
        self.avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        self.avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        # Profit factor
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        self.profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
        
        # Fees
        self.total_fees = sum(t.fees for t in self.trades)
        
        # Drawdown
        equity_curve = [self.initial_capital]
        for trade in self.trades:
            equity_curve.append(equity_curve[-1] + trade.pnl)
        
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = equity_series - running_max
        
        self.max_drawdown = abs(drawdown.min())
        self.max_drawdown_pct = (self.max_drawdown / self.initial_capital * 100) if self.initial_capital > 0 else 0
        
        # Sharpe Ratio
        returns = [t.pnl_pct for t in self.trades]
        if returns and len(returns) > 1:
            returns_mean = np.mean(returns)
            returns_std = np.std(returns)
            
            # Anualizado (assumindo ~252 dias de trading por ano)
            self.sharpe_ratio = (returns_mean / returns_std * np.sqrt(252)) if returns_std > 0 else 0
            
            # Sortino Ratio (considera apenas downside deviation)
            negative_returns = [r for r in returns if r < 0]
            if negative_returns:
                downside_std = np.std(negative_returns)
                self.sortino_ratio = (returns_mean / downside_std * np.sqrt(252)) if downside_std > 0 else 0
            else:
                self.sortino_ratio = float('inf') if returns_mean > 0 else 0
        
        # Time metrics
        if self.trades:
            self.start_date = self.trades[0].entry_time
            self.end_date = self.trades[-1].exit_time
            self.duration_days = (self.end_date - self.start_date).days
    
    def to_dict(self) -> Dict:
        """Converte resultados para dict"""
        return {
            'config': self.config.to_dict(),
            'performance': {
                'initial_capital': self.initial_capital,
                'final_capital': self.final_capital,
                'total_pnl': self.total_pnl,
                'total_pnl_pct': self.total_pnl_pct,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.win_rate,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'profit_factor': self.profit_factor,
                'max_drawdown': self.max_drawdown,
                'max_drawdown_pct': self.max_drawdown_pct,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'total_fees': self.total_fees,
                'start_date': self.start_date.strftime("%Y-%m-%d") if self.start_date else None,
                'end_date': self.end_date.strftime("%Y-%m-%d") if self.end_date else None,
                'duration_days': self.duration_days
            },
            'trades': [t.to_dict() for t in self.trades]
        }
    
    def print_summary(self):
        """Imprime resumo dos resultados"""
        logger.info("=" * 70)
        logger.info("📊 RESULTADOS DO BACKTEST")
        logger.info("=" * 70)
        logger.info(f"Símbolo: {self.config.symbol}")
        logger.info(f"Período: {self.start_date.strftime('%Y-%m-%d') if self.start_date else 'N/A'} até {self.end_date.strftime('%Y-%m-%d') if self.end_date else 'N/A'}")
        logger.info(f"Duração: {self.duration_days} dias")
        logger.info("")
        logger.info("💰 Performance:")
        logger.info(f"   Capital Inicial: ${self.initial_capital:,.2f}")
        logger.info(f"   Capital Final: ${self.final_capital:,.2f}")
        logger.info(f"   PnL Total: ${self.total_pnl:+,.2f} ({self.total_pnl_pct:+.2f}%)")
        logger.info(f"   Fees Totais: ${self.total_fees:,.2f}")
        logger.info("")
        logger.info("📈 Estatísticas:")
        logger.info(f"   Total Trades: {self.total_trades}")
        logger.info(f"   Winning Trades: {self.winning_trades}")
        logger.info(f"   Losing Trades: {self.losing_trades}")
        logger.info(f"   Win Rate: {self.win_rate:.2f}%")
        logger.info(f"   Avg Win: ${self.avg_win:,.2f}")
        logger.info(f"   Avg Loss: ${self.avg_loss:,.2f}")
        logger.info(f"   Profit Factor: {self.profit_factor:.2f}")
        logger.info("")
        logger.info("📊 Métricas de Risco:")
        logger.info(f"   Max Drawdown: ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2f}%)")
        logger.info(f"   Sharpe Ratio: {self.sharpe_ratio:.2f}")
        logger.info(f"   Sortino Ratio: {self.sortino_ratio:.2f}")
        logger.info("=" * 70)


# ============================================================================
# CLASSE PRINCIPAL: Backtester
# ============================================================================

class Backtester:
    """
    Engine de backtesting para a estratégia
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Inicializa o backtester
        
        Args:
            config: Configurações do backtest
        """
        self.config = config
        
        # Strategy
        self.strategy = ScalpingStrategy(
            atr_multiplier_sl=config.atr_multiplier_sl,
            atr_multiplier_tp1=config.atr_multiplier_tp1,
            atr_multiplier_tp2=config.atr_multiplier_tp2,
            confidence_threshold=config.confidence_threshold
        )
        
        # Risk Manager (para position sizing)
        self.risk_manager = RiskManager(
            initial_capital=config.initial_capital,
            risk_per_trade=config.risk_per_trade,
            max_positions=config.max_positions,
            leverage=config.leverage,
            trading_fee=config.trading_fee
        )
        
        # State
        self.current_position: Optional[Dict] = None
        self.trade_id_counter = 1
        self.capital = config.initial_capital
        
        logger.info(f"Backtester inicializado: {config.symbol} ({config.start_date} até {config.end_date})")
    
    # ========================================================================
    # CARREGAMENTO DE DADOS
    # ========================================================================
    
    def load_data(
        self,
        df_primary: pd.DataFrame,
        df_confirmation: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carrega e prepara dados para backtest
        
        Args:
            df_primary: DataFrame do timeframe primário
            df_confirmation: DataFrame do timeframe de confirmação
        
        Returns:
            Tupla (df_primary_filtered, df_confirmation_filtered)
        """
        logger.info("Carregando dados para backtest...")
        
        # Filtra por período
        start_dt = pd.to_datetime(self.config.start_date)
        end_dt = pd.to_datetime(self.config.end_date)
        
        df_primary = df_primary[
            (df_primary.index >= start_dt) & (df_primary.index <= end_dt)
        ].copy()
        
        df_confirmation = df_confirmation[
            (df_confirmation.index >= start_dt) & (df_confirmation.index <= end_dt)
        ].copy()
        
        logger.info(
            f"Dados carregados: "
            f"{len(df_primary)} candles ({self.config.primary_timeframe}m) | "
            f"{len(df_confirmation)} candles ({self.config.confirmation_timeframe}m)"
        )
        
        return df_primary, df_confirmation
    
    # ========================================================================
    # SIMULAÇÃO DE TRADES
    # ========================================================================
    
    def run(
        self,
        df_primary: pd.DataFrame,
        df_confirmation: pd.DataFrame
    ) -> BacktestResults:
        """
        Executa backtest
        
        Args:
            df_primary: DataFrame do timeframe primário com OHLCV
            df_confirmation: DataFrame do timeframe de confirmação
        
        Returns:
            BacktestResults com métricas e trades
        """
        logger.info("🚀 Iniciando backtest...")
        
        # Filtra dados
        df_primary, df_confirmation = self.load_data(df_primary, df_confirmation)
        
        if df_primary.empty:
            logger.error("Sem dados para backtest")
            return BacktestResults(config=self.config)
        
        # Resultados
        results = BacktestResults(
            config=self.config,
            initial_capital=self.config.initial_capital
        )
        
        # Itera pelos candles do timeframe primário
        for i in range(100, len(df_primary)):  # Começa após 100 candles (warm-up)
            current_time = df_primary.index[i]
            current_candle = df_primary.iloc[i]
            
            # Se tem posição aberta, verifica SL/TP
            if self.current_position:
                exit_info = self._check_exit(current_candle, current_time)
                
                if exit_info:
                    trade = self._close_position(exit_info, results)
                    if trade:
                        results.trades.append(trade)
                    continue
            
            # Se não tem posição, busca sinais
            if not self.current_position:
                # Pega janela de dados até o índice atual
                df_window = df_primary.iloc[:i+1]
                
                # Gera sinal
                signal = self.strategy.analyze(
                    df_window.tail(200),  # Usa últimos 200 candles
                    self.config.symbol,
                    self.config.primary_timeframe
                )
                
                if not signal:
                    continue
                
                # Confirma com timeframe maior
                df_conf_window = df_confirmation[df_confirmation.index <= current_time]
                
                if len(df_conf_window) < 50:
                    continue
                
                confirmed = self.strategy.confirm_signal_multi_tf(
                    signal_primary=signal,
                    df_higher_tf=df_conf_window.tail(100),
                    higher_tf_name=f"{self.config.confirmation_timeframe}m"
                )
                
                if not confirmed:
                    continue
                
                # Abre posição
                self._open_position(signal, current_time, current_candle)
        
        # Fecha posição aberta no final (se houver)
        if self.current_position:
            last_candle = df_primary.iloc[-1]
            exit_info = {
                'exit_time': df_primary.index[-1],
                'exit_price': last_candle['close'],
                'exit_reason': 'End of data'
            }
            trade = self._close_position(exit_info, results)
            if trade:
                results.trades.append(trade)
        
        # Calcula métricas
        results.calculate_metrics()
        
        logger.success(f"✅ Backtest concluído: {len(results.trades)} trades")
        
        return results
    
    def _open_position(
        self,
        signal: TradingSignal,
        entry_time: datetime,
        entry_candle: pd.Series
    ):
        """Abre uma posição no backtest"""
        # Calcula position size
        quantity, sizing_details = self.risk_manager.calculate_position_size(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            symbol=self.config.symbol,
            signal_type=signal.signal_type.value
        )
        
        if sizing_details.get("status") != "APPROVED":
            return
        
        # Aplica slippage
        if signal.signal_type == SignalType.LONG:
            entry_price = signal.entry_price * (1 + self.config.slippage_pct)
        else:
            entry_price = signal.entry_price * (1 - self.config.slippage_pct)
        
        # Armazena posição
        self.current_position = {
            'entry_time': entry_time,
            'entry_price': entry_price,
            'quantity': quantity,
            'side': 'Long' if signal.signal_type == SignalType.LONG else 'Short',
            'stop_loss': signal.stop_loss,
            'take_profit_1': signal.take_profit_1,
            'take_profit_2': signal.take_profit_2,
            'tp1_hit': False,
            'confidence': signal.confidence,
            'signal_reason': signal.reason
        }
        
        logger.debug(
            f"[{entry_time}] Posição aberta: {self.current_position['side']} "
            f"{quantity:.6f} @ {entry_price:.2f}"
        )
    
    def _check_exit(
        self,
        current_candle: pd.Series,
        current_time: datetime
    ) -> Optional[Dict]:
        """
        Verifica se deve sair da posição
        
        Returns:
            Dict com info de saída ou None
        """
        if not self.current_position:
            return None
        
        pos = self.current_position
        high = current_candle['high']
        low = current_candle['low']
        
        # Para LONG
        if pos['side'] == 'Long':
            # Verifica SL
            if low <= pos['stop_loss']:
                return {
                    'exit_time': current_time,
                    'exit_price': pos['stop_loss'],
                    'exit_reason': 'SL'
                }
            
            # Verifica TP1
            if not pos['tp1_hit'] and high >= pos['take_profit_1']:
                pos['tp1_hit'] = True
                pos['stop_loss'] = pos['entry_price']  # Move SL para breakeven
                
                return {
                    'exit_time': current_time,
                    'exit_price': pos['take_profit_1'],
                    'exit_reason': 'TP1',
                    'partial': True,
                    'partial_pct': 0.5
                }
            
            # Verifica TP2
            if pos['tp1_hit'] and high >= pos['take_profit_2']:
                return {
                    'exit_time': current_time,
                    'exit_price': pos['take_profit_2'],
                    'exit_reason': 'TP2'
                }
        
        # Para SHORT
        else:
            # Verifica SL
            if high >= pos['stop_loss']:
                return {
                    'exit_time': current_time,
                    'exit_price': pos['stop_loss'],
                    'exit_reason': 'SL'
                }
            
            # Verifica TP1
            if not pos['tp1_hit'] and low <= pos['take_profit_1']:
                pos['tp1_hit'] = True
                pos['stop_loss'] = pos['entry_price']  # Move SL para breakeven
                
                return {
                    'exit_time': current_time,
                    'exit_price': pos['take_profit_1'],
                    'exit_reason': 'TP1',
                    'partial': True,
                    'partial_pct': 0.5
                }
            
            # Verifica TP2
            if pos['tp1_hit'] and low <= pos['take_profit_2']:
                return {
                    'exit_time': current_time,
                    'exit_price': pos['take_profit_2'],
                    'exit_reason': 'TP2'
                }
        
        return None
    
    def _close_position(
        self,
        exit_info: Dict,
        results: BacktestResults
    ) -> Optional[BacktestTrade]:
        """Fecha posição e registra trade"""
        if not self.current_position:
            return None
        
        pos = self.current_position
        
        # Determina quantidade a fechar
        if exit_info.get('partial'):
            qty_to_close = pos['quantity'] * exit_info['partial_pct']
        else:
            qty_to_close = pos['quantity']
        
        # Calcula PnL
        if pos['side'] == 'Long':
            pnl = (exit_info['exit_price'] - pos['entry_price']) * qty_to_close
            pnl_pct = ((exit_info['exit_price'] - pos['entry_price']) / pos['entry_price']) * 100
        else:
            pnl = (pos['entry_price'] - exit_info['exit_price']) * qty_to_close
            pnl_pct = ((pos['entry_price'] - exit_info['exit_price']) / pos['entry_price']) * 100
        
        # Aplica alavancagem ao PnL percentual
        pnl_pct_leveraged = pnl_pct * self.config.leverage
        
        # Calcula fees
        entry_value = pos['entry_price'] * qty_to_close
        exit_value = exit_info['exit_price'] * qty_to_close
        fees = (entry_value + exit_value) * self.config.trading_fee
        
        # PnL líquido
        net_pnl = pnl - fees
        
        # Atualiza capital
        self.capital += net_pnl
        
        # Cria registro do trade
        trade = BacktestTrade(
            trade_id=self.trade_id_counter,
            entry_time=pos['entry_time'],
            exit_time=exit_info['exit_time'],
            symbol=self.config.symbol,
            side=pos['side'],
            entry_price=pos['entry_price'],
            exit_price=exit_info['exit_price'],
            quantity=qty_to_close,
            leverage=self.config.leverage,
            stop_loss=pos['stop_loss'],
            take_profit_1=pos['take_profit_1'],
            take_profit_2=pos['take_profit_2'],
            exit_reason=exit_info['exit_reason'],
            pnl=net_pnl,
            pnl_pct=pnl_pct_leveraged,
            fees=fees,
            confidence=pos['confidence'],
            signal_reason=pos['signal_reason']
        )
        
        self.trade_id_counter += 1
        
        logger.debug(
            f"[{exit_info['exit_time']}] Posição fechada: "
            f"PnL = ${net_pnl:+.2f} ({pnl_pct_leveraged:+.2f}%) | "
            f"Razão: {exit_info['exit_reason']}"
        )
        
        # Se foi fechamento parcial, atualiza posição
        if exit_info.get('partial'):
            pos['quantity'] -= qty_to_close
        else:
            self.current_position = None
        
        return trade
    
    # ========================================================================
    # EXPORTAÇÃO DE RESULTADOS
    # ========================================================================
    
    def export_results(
        self,
        results: BacktestResults,
        output_dir: str = "backtest_results"
    ):
        """
        Exporta resultados do backtest
        
        Args:
            results: Resultados do backtest
            output_dir: Diretório de saída
        """
        import os
        
        # Cria diretório
        os.makedirs(output_dir, exist_ok=True)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol = self.config.symbol
        
        # 1. JSON completo
        json_file = f"{output_dir}/backtest_{symbol}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"📁 JSON exportado: {json_file}")
        
        # 2. CSV de trades
        if results.trades:
            trades_data = [t.to_dict() for t in results.trades]
            df_trades = pd.DataFrame(trades_data)
            csv_file = f"{output_dir}/trades_{symbol}_{timestamp}.csv"
            df_trades.to_csv(csv_file, index=False)
            logger.info(f"📁 CSV exportado: {csv_file}")
        
        # 3. Relatório TXT
        txt_file = f"{output_dir}/report_{symbol}_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("RELATÓRIO DE BACKTEST\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Símbolo: {self.config.symbol}\n")
            f.write(f"Período: {results.start_date.strftime('%Y-%m-%d') if results.start_date else 'N/A'} até {results.end_date.strftime('%Y-%m-%d') if results.end_date else 'N/A'}\n")
            f.write(f"Duração: {results.duration_days} dias\n\n")
            
            f.write("PERFORMANCE:\n")
            f.write(f"  Capital Inicial: ${results.initial_capital:,.2f}\n")
            f.write(f"  Capital Final: ${results.final_capital:,.2f}\n")
            f.write(f"  PnL Total: ${results.total_pnl:+,.2f} ({results.total_pnl_pct:+.2f}%)\n\n")
            
            f.write("ESTATÍSTICAS:\n")
            f.write(f"  Total Trades: {results.total_trades}\n")
            f.write(f"  Win Rate: {results.win_rate:.2f}%\n")
            f.write(f"  Profit Factor: {results.profit_factor:.2f}\n")
            f.write(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}\n")
            f.write(f"  Max Drawdown: ${results.max_drawdown:,.2f} ({results.max_drawdown_pct:.2f}%)\n")
        
        logger.info(f"📁 Relatório exportado: {txt_file}")


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Exemplo de uso do backtester
    """
    from loguru import logger
    
    # Configuração de log
    logger.add(
        "logs/backtest.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )
    
    # Configuração do backtest
    config = BacktestConfig(
        symbol="BTCUSDT",
        start_date="2024-01-01",
        end_date="2024-06-30",
        initial_capital=10000.0,
        risk_per_trade=0.01,
        leverage=5.0,
        confidence_threshold=0.65
    )
    
    # Dados fictícios (substituir por dados reais)
    dates_15m = pd.date_range('2024-01-01', '2024-06-30', freq='15min')
    dates_1h = pd.date_range('2024-01-01', '2024-06-30', freq='1H')
    
    # Simula preços
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
    
    # Cria backtester
    backtester = Backtester(config)
    
    # Executa backtest
    results = backtester.run(df_15m, df_1h)
    
    # Imprime resultados
    results.print_summary()
    
    # Exporta resultados
    backtester.export_results(results)