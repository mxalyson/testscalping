"""
Risk Manager - Gestão de Risco e Position Sizing
================================================

Módulo responsável por:
- Cálculo de tamanho de posição baseado em risco percentual
- Gestão de alavancagem
- Circuit breaker por drawdown diário
- Limites de exposição
- Validação de ordens
- Tracking de performance

Autor: Trading Bot Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger
import json


# ============================================================================
# ENUMS E CONSTANTES
# ============================================================================

class RiskStatus(Enum):
    """Status do gerenciador de risco"""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    MAX_POSITIONS = "MAX_POSITIONS"
    MAX_LEVERAGE = "MAX_LEVERAGE"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Position:
    """Representa uma posição aberta"""
    symbol: str
    side: str  # "Long" ou "Short"
    entry_price: float
    quantity: float
    leverage: float
    entry_time: int
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    tp1_hit: bool = False
    sl_moved_to_breakeven: bool = False
    
    def update_pnl(self, current_price: float):
        """Atualiza PnL não realizado"""
        self.current_price = current_price
        
        if self.side == "Long":
            pnl = (current_price - self.entry_price) * self.quantity
            pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # Short
            pnl = (self.entry_price - current_price) * self.quantity
            pnl_pct = ((self.entry_price - current_price) / self.entry_price) * 100
        
        # Aplica alavancagem ao PnL percentual
        self.unrealized_pnl = pnl
        self.unrealized_pnl_pct = pnl_pct * self.leverage
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'leverage': self.leverage,
            'entry_time': self.entry_time,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'tp1_hit': self.tp1_hit,
            'sl_moved_to_breakeven': self.sl_moved_to_breakeven
        }


@dataclass
class TradeRecord:
    """Registro de um trade fechado"""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: float
    entry_time: int
    exit_time: int
    pnl: float
    pnl_pct: float
    fees: float
    reason: str  # "TP1", "TP2", "SL", "Manual"
    
    def to_dict(self) -> Dict:
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'leverage': self.leverage,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'fees': self.fees,
            'reason': self.reason
        }


# ============================================================================
# CLASSE PRINCIPAL: RiskManager
# ============================================================================

class RiskManager:
    """
    Gerenciador de risco para o bot de trading
    """
    
    def __init__(
        self,
        initial_capital: float,
        risk_per_trade: float = 0.01,  # 1% do capital por trade
        max_positions: int = 2,
        leverage: float = 5.0,
        max_daily_drawdown: float = 0.05,  # 5%
        max_total_drawdown: float = 0.20,  # 20%
        circuit_breaker_enabled: bool = True,
        trading_fee: float = 0.0006,  # 0.06% maker/taker
        min_position_size_usdt: float = 10.0
    ):
        """
        Inicializa o Risk Manager
        
        Args:
            initial_capital: Capital inicial em USDT
            risk_per_trade: Percentual do capital arriscado por trade
            max_positions: Número máximo de posições simultâneas
            leverage: Alavancagem padrão
            max_daily_drawdown: Drawdown máximo diário antes de circuit breaker
            max_total_drawdown: Drawdown máximo total
            circuit_breaker_enabled: Se True, ativa circuit breaker
            trading_fee: Taxa de trading (maker/taker)
            min_position_size_usdt: Tamanho mínimo de posição em USDT
        """
        # Capital
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        # Parâmetros de risco
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.leverage = leverage
        self.max_daily_drawdown = max_daily_drawdown
        self.max_total_drawdown = max_total_drawdown
        self.circuit_breaker_enabled = circuit_breaker_enabled
        self.trading_fee = trading_fee
        self.min_position_size_usdt = min_position_size_usdt
        
        # Status
        self.status = RiskStatus.NORMAL
        self.circuit_breaker_active = False
        self.circuit_breaker_until: Optional[datetime] = None
        
        # Posições
        self.positions: Dict[str, Position] = {}
        
        # Histórico de trades
        self.trade_history: List[TradeRecord] = []
        self.next_trade_id = 1
        
        # Métricas diárias
        self.daily_pnl = 0.0
        self.daily_start_capital = initial_capital
        self.current_day = datetime.now().date()
        
        # Estatísticas
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = 0.0
        
        logger.info(
            f"RiskManager inicializado: Capital={initial_capital:.2f} USDT, "
            f"Risk={risk_per_trade*100:.1f}%, Max Positions={max_positions}, "
            f"Leverage={leverage}x"
        )
    
    # ========================================================================
    # POSITION SIZING
    # ========================================================================
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        symbol: str,
        signal_type: str
    ) -> Tuple[float, Dict]:
        """
        Calcula tamanho da posição baseado no risco
        
        Args:
            entry_price: Preço de entrada
            stop_loss: Preço de stop loss
            symbol: Símbolo do ativo
            signal_type: "LONG" ou "SHORT"
        
        Returns:
            (quantidade, dict com detalhes)
        """
        # Verifica circuit breaker
        if self.circuit_breaker_active:
            return 0.0, {
                "error": "Circuit breaker ativo",
                "status": "REJECTED"
            }
        
        # Verifica limite de posições
        if len(self.positions) >= self.max_positions:
            return 0.0, {
                "error": f"Máximo de {self.max_positions} posições atingido",
                "status": "REJECTED"
            }
        
        # Calcula risco em USDT
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Calcula distância até SL
        if signal_type == "LONG":
            distance_to_sl = entry_price - stop_loss
        else:  # SHORT
            distance_to_sl = stop_loss - entry_price
        
        if distance_to_sl <= 0:
            return 0.0, {
                "error": "Stop loss inválido",
                "status": "REJECTED"
            }
        
        # Distância percentual
        distance_pct = distance_to_sl / entry_price
        
        # Calcula quantidade
        # Fórmula: quantity = (risk_amount / distance_to_sl) / leverage
        # Simplificando: quantity = risk_amount / (distance_pct * entry_price)
        position_size_usdt = risk_amount / distance_pct
        quantity = position_size_usdt / entry_price
        
        # Ajusta pela alavancagem (capital necessário)
        required_margin = position_size_usdt / self.leverage
        
        # Verifica se tem capital suficiente
        if required_margin > self.current_capital * 0.9:  # Usa no máx 90% do capital
            return 0.0, {
                "error": f"Capital insuficiente. Necessário: {required_margin:.2f}, Disponível: {self.current_capital:.2f}",
                "status": "REJECTED"
            }
        
        # Verifica tamanho mínimo
        if position_size_usdt < self.min_position_size_usdt:
            return 0.0, {
                "error": f"Posição muito pequena. Mínimo: {self.min_position_size_usdt} USDT",
                "status": "REJECTED"
            }
        
        # Calcula fees estimadas
        estimated_fees = position_size_usdt * self.trading_fee * 2  # Entry + Exit
        
        details = {
            "status": "APPROVED",
            "quantity": quantity,
            "position_size_usdt": position_size_usdt,
            "required_margin": required_margin,
            "risk_amount": risk_amount,
            "risk_pct": self.risk_per_trade * 100,
            "distance_to_sl_pct": distance_pct * 100,
            "leverage": self.leverage,
            "estimated_fees": estimated_fees
        }
        
        logger.info(
            f"📊 Position Size calculado: {symbol} {signal_type} | "
            f"Qty: {quantity:.6f} | Size: {position_size_usdt:.2f} USDT | "
            f"Margin: {required_margin:.2f} USDT | Risk: {risk_amount:.2f} USDT"
        )
        
        return quantity, details
    
    # ========================================================================
    # VALIDAÇÃO DE TRADES
    # ========================================================================
    
    def validate_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float
    ) -> Tuple[bool, str]:
        """
        Valida se um trade pode ser executado
        
        Returns:
            (is_valid, reason)
        """
        # Circuit breaker
        if self.circuit_breaker_active:
            return False, "Circuit breaker ativo"
        
        # Máximo de posições
        if len(self.positions) >= self.max_positions:
            return False, f"Máximo de {self.max_positions} posições"
        
        # Já tem posição neste símbolo
        if symbol in self.positions:
            return False, f"Já existe posição aberta em {symbol}"
        
        # Valida preços
        if quantity <= 0:
            return False, "Quantidade inválida"
        
        if entry_price <= 0:
            return False, "Preço de entrada inválido"
        
        # Valida SL e TP para LONG
        if side == "Long":
            if stop_loss >= entry_price:
                return False, "Stop loss deve estar abaixo do preço de entrada (LONG)"
            if take_profit_1 <= entry_price or take_profit_2 <= entry_price:
                return False, "Take profits devem estar acima do preço de entrada (LONG)"
        
        # Valida SL e TP para SHORT
        elif side == "Short":
            if stop_loss <= entry_price:
                return False, "Stop loss deve estar acima do preço de entrada (SHORT)"
            if take_profit_1 >= entry_price or take_profit_2 >= entry_price:
                return False, "Take profits devem estar abaixo do preço de entrada (SHORT)"
        
        return True, "Trade válido"
    
    # ========================================================================
    # GERENCIAMENTO DE POSIÇÕES
    # ========================================================================
    
    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float
    ) -> bool:
        """
        Abre uma nova posição
        
        Returns:
            True se sucesso, False se falhou
        """
        # Valida trade
        is_valid, reason = self.validate_trade(
            symbol, side, quantity, entry_price,
            stop_loss, take_profit_1, take_profit_2
        )
        
        if not is_valid:
            logger.warning(f"❌ Trade rejeitado: {reason}")
            return False
        
        # Cria posição
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            leverage=self.leverage,
            entry_time=int(datetime.now().timestamp() * 1000),
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2
        )
        
        # Adiciona às posições ativas
        self.positions[symbol] = position
        
        # Calcula fees de entrada
        entry_fee = (entry_price * quantity) * self.trading_fee
        self.total_fees += entry_fee
        
        logger.success(
            f"✅ Posição aberta: {side} {quantity:.6f} {symbol} @ {entry_price:.2f} | "
            f"SL: {stop_loss:.2f} | TP1: {take_profit_1:.2f} | TP2: {take_profit_2:.2f}"
        )
        
        return True
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        quantity: Optional[float] = None,
        reason: str = "Manual"
    ) -> Optional[TradeRecord]:
        """
        Fecha uma posição (total ou parcialmente)
        
        Args:
            symbol: Símbolo da posição
            exit_price: Preço de saída
            quantity: Quantidade a fechar (None = total)
            reason: Razão do fechamento
        
        Returns:
            TradeRecord do trade fechado ou None
        """
        if symbol not in self.positions:
            logger.warning(f"Posição não encontrada: {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # Se não especificou quantidade, fecha tudo
        if quantity is None:
            quantity = position.quantity
        
        # Não pode fechar mais do que tem
        quantity = min(quantity, position.quantity)
        
        # Calcula PnL
        if position.side == "Long":
            pnl = (exit_price - position.entry_price) * quantity
            pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
        else:  # Short
            pnl = (position.entry_price - exit_price) * quantity
            pnl_pct = ((position.entry_price - exit_price) / position.entry_price) * 100
        
        # Aplica alavancagem ao PnL percentual
        pnl_pct_leveraged = pnl_pct * position.leverage
        
        # Calcula fees
        exit_fee = (exit_price * quantity) * self.trading_fee
        total_fees = exit_fee
        
        # PnL líquido
        net_pnl = pnl - total_fees
        
        # Atualiza capital
        self.current_capital += net_pnl
        self.daily_pnl += net_pnl
        self.total_fees += total_fees
        
        # Atualiza peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Cria registro do trade
        trade_id = f"TRADE_{self.next_trade_id:06d}"
        self.next_trade_id += 1
        
        trade_record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=quantity,
            leverage=position.leverage,
            entry_time=position.entry_time,
            exit_time=int(datetime.now().timestamp() * 1000),
            pnl=net_pnl,
            pnl_pct=pnl_pct_leveraged,
            fees=total_fees,
            reason=reason
        )
        
        # Atualiza estatísticas
        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Adiciona ao histórico
        self.trade_history.append(trade_record)
        
        # Se fechou tudo, remove a posição
        if quantity >= position.quantity:
            del self.positions[symbol]
            logger.success(
                f"💰 Posição fechada: {symbol} | "
                f"PnL: {net_pnl:+.2f} USDT ({pnl_pct_leveraged:+.2f}%) | "
                f"Razão: {reason}"
            )
        else:
            # Fechamento parcial
            position.quantity -= quantity
            logger.info(
                f"💰 Fechamento parcial: {symbol} | "
                f"Qty fechada: {quantity:.6f} | "
                f"PnL: {net_pnl:+.2f} USDT ({pnl_pct_leveraged:+.2f}%) | "
                f"Razão: {reason}"
            )
        
        # Verifica circuit breaker
        self._check_circuit_breaker()
        
        return trade_record
    
    def update_stop_loss(self, symbol: str, new_sl: float):
        """Atualiza stop loss de uma posição"""
        if symbol not in self.positions:
            return
        
        old_sl = self.positions[symbol].stop_loss
        self.positions[symbol].stop_loss = new_sl
        
        logger.info(
            f"🔄 Stop Loss atualizado: {symbol} | "
            f"{old_sl:.2f} → {new_sl:.2f}"
        )
    
    def move_sl_to_breakeven(self, symbol: str):
        """Move stop loss para breakeven"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        if not position.sl_moved_to_breakeven:
            self.update_stop_loss(symbol, position.entry_price)
            position.sl_moved_to_breakeven = True
            
            logger.success(
                f"🎯 SL movido para breakeven: {symbol} @ {position.entry_price:.2f}"
            )
    
    def mark_tp1_hit(self, symbol: str):
        """Marca que TP1 foi atingido"""
        if symbol in self.positions:
            self.positions[symbol].tp1_hit = True
            logger.info(f"✅ TP1 atingido: {symbol}")
    
    # ========================================================================
    # MONITORAMENTO E CIRCUIT BREAKER
    # ========================================================================
    
    def update_positions(self, prices: Dict[str, float]):
        """
        Atualiza PnL de todas as posições
        
        Args:
            prices: Dict {symbol: current_price}
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_pnl(prices[symbol])
    
    def _check_circuit_breaker(self):
        """Verifica se deve ativar circuit breaker"""
        if not self.circuit_breaker_enabled:
            return
        
        # Verifica novo dia
        today = datetime.now().date()
        if today != self.current_day:
            self.current_day = today
            self.daily_start_capital = self.current_capital
            self.daily_pnl = 0.0
            
            # Desativa circuit breaker se estava ativo
            if self.circuit_breaker_active:
                self.circuit_breaker_active = False
                self.circuit_breaker_until = None
                logger.success("✅ Circuit breaker desativado (novo dia)")
        
        # Calcula drawdown diário
        daily_dd = (self.daily_start_capital - self.current_capital) / self.daily_start_capital
        
        # Calcula drawdown total
        total_dd = (self.peak_capital - self.current_capital) / self.peak_capital
        
        # Ativa circuit breaker se necessário
        if daily_dd >= self.max_daily_drawdown:
            self.circuit_breaker_active = True
            self.circuit_breaker_until = datetime.now() + timedelta(days=1)
            self.status = RiskStatus.CIRCUIT_BREAKER
            
            logger.critical(
                f"🚨 CIRCUIT BREAKER ATIVADO! "
                f"Drawdown diário: {daily_dd*100:.2f}% "
                f"(limite: {self.max_daily_drawdown*100:.1f}%)"
            )
        
        elif total_dd >= self.max_total_drawdown:
            self.circuit_breaker_active = True
            self.status = RiskStatus.CIRCUIT_BREAKER
            
            logger.critical(
                f"🚨 CIRCUIT BREAKER ATIVADO! "
                f"Drawdown total: {total_dd*100:.2f}% "
                f"(limite: {self.max_total_drawdown*100:.1f}%)"
            )
    
    def get_risk_status(self) -> Dict:
        """Retorna status atual do gerenciador de risco"""
        # Calcula drawdowns
        daily_dd = 0.0
        if self.daily_start_capital > 0:
            daily_dd = (self.daily_start_capital - self.current_capital) / self.daily_start_capital
        
        total_dd = 0.0
        if self.peak_capital > 0:
            total_dd = (self.peak_capital - self.current_capital) / self.peak_capital
        
        # Calcula PnL total
        total_pnl = self.current_capital - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        # Win rate
        win_rate = 0.0
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
        
        return {
            "status": self.status.value,
            "circuit_breaker_active": self.circuit_breaker_active,
            "current_capital": self.current_capital,
            "initial_capital": self.initial_capital,
            "peak_capital": self.peak_capital,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "daily_pnl": self.daily_pnl,
            "daily_drawdown_pct": daily_dd * 100,
            "total_drawdown_pct": total_dd * 100,
            "max_daily_drawdown_pct": self.max_daily_drawdown * 100,
            "max_total_drawdown_pct": self.max_total_drawdown * 100,
            "open_positions": len(self.positions),
            "max_positions": self.max_positions,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "total_fees": self.total_fees
        }
    
    # ========================================================================
    # RELATÓRIOS E ESTATÍSTICAS
    # ========================================================================
    
    def get_performance_metrics(self) -> Dict:
        """Calcula métricas de performance"""
        if not self.trade_history:
            return {}
        
        # Converte trades para DataFrame
        trades_data = [t.to_dict() for t in self.trade_history]
        df = pd.DataFrame(trades_data)
        
        # Calcula métricas
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        win_rate = (len(winning_trades) / len(df)) * 100 if len(df) > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
        
        # Sharpe Ratio simplificado
        returns = df['pnl_pct'].values
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown na sequência de trades
        cumulative_pnl = df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_dd = drawdown.min()
        
        return {
            "total_trades": len(df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_fees": self.total_fees
        }
    
    def export_trade_history(self, filename: str = "trade_history.csv"):
        """Exporta histórico de trades para CSV"""
        if not self.trade_history:
            logger.warning("Nenhum trade no histórico")
            return
        
        trades_data = [t.to_dict() for t in self.trade_history]
        df = pd.DataFrame(trades_data)
        
        df.to_csv(filename, index=False)
        logger.success(f"📁 Histórico exportado: {filename}")
    
    def __repr__(self) -> str:
        return (
            f"RiskManager("
            f"capital={self.current_capital:.2f}, "
            f"positions={len(self.positions)}, "
            f"status={self.status.value})"
        )


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Exemplo de uso do Risk Manager
    """
    
    # Configuração de log
    logger.add(
        "logs/risk_manager.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )
    
    # Inicializa
    rm = RiskManager(
        initial_capital=10000.0,
        risk_per_trade=0.01,
        max_positions=2,
        leverage=5.0,
        max_daily_drawdown=0.05
    )
    
    # Exemplo: Calcula position size
    qty, details = rm.calculate_position_size(
        entry_price=42000.0,
        stop_loss=41500.0,
        symbol="BTCUSDT",
        signal_type="LONG"
    )
    
    logger.info(f"Position size: {qty:.6f} BTC")
    logger.info(f"Detalhes: {json.dumps(details, indent=2)}")
    
    # Abre posição
    if qty > 0:
        success = rm.open_position(
            symbol="BTCUSDT",
            side="Long",
            entry_price=42000.0,
            quantity=qty,
            stop_loss=41500.0,
            take_profit_1=42500.0,
            take_profit_2=43000.0
        )
        
        if success:
            logger.success("Posição aberta com sucesso")
            
            # Simula atualização de preço
            rm.update_positions({"BTCUSDT": 42300.0})
            
            # Move SL para breakeven
            rm.move_sl_to_breakeven("BTCUSDT")
            
            # Fecha 50% em TP1
            rm.mark_tp1_hit("BTCUSDT")
            rm.close_position("BTCUSDT", 42500.0, quantity=qty * 0.5, reason="TP1")
            
            # Fecha restante em TP2
            rm.close_position("BTCUSDT", 43000.0, reason="TP2")
    
    # Status final
    status = rm.get_risk_status()
    logger.info(f"\n📊 Status Final:\n{json.dumps(status, indent=2)}")
    
    # Métricas
    metrics = rm.get_performance_metrics()
    logger.info(f"\n📈 Métricas:\n{json.dumps(metrics, indent=2)}")