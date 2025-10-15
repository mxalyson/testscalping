"""
Order Manager - Gestão de Ordens e Execução
===========================================

Módulo responsável por:
- Envio de ordens (market/limit) em paper mode ou real
- Monitoramento de posições abertas
- Gerenciamento de Stop Loss e Take Profit
- Fechamento parcial em TP1 e TP2
- Movimento de SL para breakeven
- Cancelamento de ordens pendentes
- Integração com Risk Manager e WebSocket

Autor: Trading Bot Team
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import threading
from datetime import datetime


# ============================================================================
# ENUMS E CONSTANTES
# ============================================================================

class OrderStatus(Enum):
    """Status de uma ordem"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(Enum):
    """Lado da ordem"""
    BUY = "Buy"
    SELL = "Sell"


class OrderType(Enum):
    """Tipo de ordem"""
    MARKET = "Market"
    LIMIT = "Limit"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Order:
    """Representa uma ordem"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    filled_qty: float = 0.0
    filled_price: Optional[float] = None
    created_at: int = 0
    updated_at: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status.value,
            'filled_qty': self.filled_qty,
            'filled_price': self.filled_price,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


@dataclass
class ManagedPosition:
    """
    Representa uma posição gerenciada com ordens de SL/TP
    """
    symbol: str
    side: str  # "Long" ou "Short"
    entry_price: float
    quantity: float
    quantity_remaining: float  # Quantidade ainda aberta
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    tp1_hit: bool = False
    sl_moved_to_breakeven: bool = False
    entry_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    tp1_order_id: Optional[str] = None
    tp2_order_id: Optional[str] = None
    created_at: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'quantity_remaining': self.quantity_remaining,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'tp1_hit': self.tp1_hit,
            'sl_moved_to_breakeven': self.sl_moved_to_breakeven,
            'entry_order_id': self.entry_order_id,
            'sl_order_id': self.sl_order_id,
            'tp1_order_id': self.tp1_order_id,
            'tp2_order_id': self.tp2_order_id,
            'created_at': self.created_at
        }


# ============================================================================
# CLASSE PRINCIPAL: OrderManager
# ============================================================================

class OrderManager:
    """
    Gerenciador de ordens e execução de trades
    """
    
    def __init__(
        self,
        ws_client,  # BybitWebSocket instance
        risk_manager,  # RiskManager instance
        paper_mode: bool = True,
        slippage_pct: float = 0.001,  # 0.1% de slippage
        partial_tp1_pct: float = 0.5,  # Fecha 50% em TP1
        enable_trailing_sl: bool = False,
        trailing_sl_pct: float = 0.005  # 0.5% trailing
    ):
        """
        Inicializa o Order Manager
        
        Args:
            ws_client: Instância do BybitWebSocket
            risk_manager: Instância do RiskManager
            paper_mode: Se True, simula ordens
            slippage_pct: Slippage estimado em %
            partial_tp1_pct: Percentual da posição a fechar em TP1
            enable_trailing_sl: Se True, ativa trailing stop loss
            trailing_sl_pct: Percentual do trailing SL
        """
        self.ws_client = ws_client
        self.risk_manager = risk_manager
        self.paper_mode = paper_mode
        self.slippage_pct = slippage_pct
        self.partial_tp1_pct = partial_tp1_pct
        self.enable_trailing_sl = enable_trailing_sl
        self.trailing_sl_pct = trailing_sl_pct
        
        # Ordens pendentes e histórico
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
        # Posições gerenciadas
        self.managed_positions: Dict[str, ManagedPosition] = {}
        
        # Lock para thread safety
        self.lock = threading.RLock()
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Contadores
        self.next_order_id = 1
        
        logger.info(
            f"OrderManager inicializado - "
            f"Mode: {'PAPER' if paper_mode else 'REAL'} | "
            f"Slippage: {slippage_pct*100:.2f}% | "
            f"TP1 Close: {partial_tp1_pct*100:.0f}%"
        )
    
    # ========================================================================
    # ENVIO DE ORDENS
    # ========================================================================
    
    def place_market_order(
        self,
        symbol: str,
        side: str,  # "Buy" ou "Sell"
        quantity: float
    ) -> Optional[Dict]:
        """
        Envia ordem a mercado
        
        Args:
            symbol: Símbolo do ativo
            side: "Buy" ou "Sell"
            quantity: Quantidade
        
        Returns:
            Dict com detalhes da ordem ou None se falhou
        """
        logger.info(f"📤 Enviando ordem MARKET: {side} {quantity:.6f} {symbol}")
        
        # Valida quantidade
        if quantity <= 0:
            logger.error("Quantidade inválida")
            return None
        
        # Pega preço atual
        current_price = self._get_current_price(symbol)
        if not current_price:
            logger.error(f"Não foi possível obter preço de {symbol}")
            return None
        
        # Aplica slippage
        if side == "Buy":
            execution_price = current_price * (1 + self.slippage_pct)
        else:
            execution_price = current_price * (1 - self.slippage_pct)
        
        # Cria ordem
        order_id = self._generate_order_id()
        timestamp = int(time.time() * 1000)
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=OrderSide.BUY if side == "Buy" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=execution_price,
            status=OrderStatus.FILLED,
            filled_qty=quantity,
            filled_price=execution_price,
            created_at=timestamp,
            updated_at=timestamp
        )
        
        # Envia ordem
        if self.paper_mode:
            # Paper mode: simula
            result = self._execute_paper_order(order)
        else:
            # Real mode: envia para exchange
            result = self._execute_real_order(order)
        
        if result and result.get("success"):
            with self.lock:
                self.order_history.append(order)
            
            logger.success(
                f"✅ Ordem MARKET executada: {order_id} | "
                f"{side} {quantity:.6f} {symbol} @ {execution_price:.2f}"
            )
            
            return {
                "success": True,
                "order_id": order_id,
                "filled_price": execution_price,
                "filled_qty": quantity,
                "order": order.to_dict()
            }
        else:
            logger.error(f"❌ Falha ao executar ordem: {result.get('error', 'Unknown')}")
            return None
    
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Optional[Dict]:
        """
        Envia ordem limit
        
        Args:
            symbol: Símbolo do ativo
            side: "Buy" ou "Sell"
            quantity: Quantidade
            price: Preço limite
        
        Returns:
            Dict com detalhes da ordem ou None
        """
        logger.info(
            f"📤 Enviando ordem LIMIT: {side} {quantity:.6f} {symbol} @ {price:.2f}"
        )
        
        order_id = self._generate_order_id()
        timestamp = int(time.time() * 1000)
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=OrderSide.BUY if side == "Buy" else OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            created_at=timestamp,
            updated_at=timestamp
        )
        
        with self.lock:
            self.pending_orders[order_id] = order
        
        logger.info(f"✅ Ordem LIMIT criada: {order_id}")
        
        return {
            "success": True,
            "order_id": order_id,
            "order": order.to_dict()
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancela uma ordem pendente"""
        with self.lock:
            if order_id not in self.pending_orders:
                logger.warning(f"Ordem não encontrada: {order_id}")
                return False
            
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            order.updated_at = int(time.time() * 1000)
            
            # Move para histórico
            self.order_history.append(order)
            del self.pending_orders[order_id]
        
        logger.info(f"❌ Ordem cancelada: {order_id}")
        return True
    
    # ========================================================================
    # ABERTURA DE POSIÇÕES COM SL/TP
    # ========================================================================
    
    def open_position_with_targets(
        self,
        symbol: str,
        side: str,  # "Long" ou "Short"
        quantity: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float
    ) -> Optional[ManagedPosition]:
        """
        Abre posição com SL e TPs automáticos
        
        Args:
            symbol: Símbolo
            side: "Long" ou "Short"
            quantity: Quantidade
            stop_loss: Preço de stop loss
            take_profit_1: Primeiro take profit (fecha 50%)
            take_profit_2: Segundo take profit (fecha restante)
        
        Returns:
            ManagedPosition criada ou None
        """
        logger.info(
            f"🎯 Abrindo posição: {side} {quantity:.6f} {symbol} | "
            f"SL: {stop_loss:.2f} | TP1: {take_profit_1:.2f} | TP2: {take_profit_2:.2f}"
        )
        
        # Valida com risk manager
        is_valid, reason = self.risk_manager.validate_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=self._get_current_price(symbol) or 0,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2
        )
        
        if not is_valid:
            logger.error(f"❌ Trade inválido: {reason}")
            return None
        
        # Envia ordem de entrada (market)
        entry_side = "Buy" if side == "Long" else "Sell"
        entry_result = self.place_market_order(symbol, entry_side, quantity)
        
        if not entry_result or not entry_result.get("success"):
            logger.error("❌ Falha ao enviar ordem de entrada")
            return None
        
        entry_price = entry_result["filled_price"]
        entry_order_id = entry_result["order_id"]
        
        # Registra posição no risk manager
        success = self.risk_manager.open_position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2
        )
        
        if not success:
            logger.error("❌ Falha ao registrar posição no risk manager")
            return None
        
        # Cria posição gerenciada
        managed_pos = ManagedPosition(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            quantity_remaining=quantity,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            entry_order_id=entry_order_id,
            created_at=int(time.time() * 1000)
        )
        
        with self.lock:
            self.managed_positions[symbol] = managed_pos
        
        logger.success(
            f"✅ Posição aberta: {side} {quantity:.6f} {symbol} @ {entry_price:.2f}"
        )
        
        return managed_pos
    
    # ========================================================================
    # MONITORAMENTO DE POSIÇÕES
    # ========================================================================
    
    def start_monitoring(self):
        """Inicia monitoramento de posições em thread separada"""
        if self.monitoring_active:
            logger.warning("Monitoramento já está ativo")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("🔍 Monitoramento de posições iniciado")
    
    def stop_monitoring(self):
        """Para o monitoramento"""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("🛑 Monitoramento de posições parado")
    
    def _monitoring_loop(self):
        """Loop de monitoramento de posições"""
        while self.monitoring_active:
            try:
                with self.lock:
                    positions_to_check = list(self.managed_positions.values())
                
                for position in positions_to_check:
                    self._check_position_targets(position)
                
                time.sleep(1)  # Checa a cada 1 segundo
            
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
                time.sleep(5)
    
    def _check_position_targets(self, position: ManagedPosition):
        """
        Verifica se posição atingiu SL ou TPs
        
        Args:
            position: Posição gerenciada
        """
        symbol = position.symbol
        current_price = self._get_current_price(symbol)
        
        if not current_price:
            return
        
        # Atualiza PnL no risk manager
        self.risk_manager.update_positions({symbol: current_price})
        
        # ====================================================================
        # VERIFICA STOP LOSS
        # ====================================================================
        
        sl_hit = False
        
        if position.side == "Long":
            sl_hit = current_price <= position.stop_loss
        else:  # Short
            sl_hit = current_price >= position.stop_loss
        
        if sl_hit:
            logger.warning(
                f"🛑 STOP LOSS atingido: {symbol} @ {current_price:.2f} "
                f"(SL: {position.stop_loss:.2f})"
            )
            self._close_position_at_market(position, "STOP_LOSS")
            return
        
        # ====================================================================
        # VERIFICA TAKE PROFIT 1
        # ====================================================================
        
        tp1_hit = False
        
        if not position.tp1_hit:
            if position.side == "Long":
                tp1_hit = current_price >= position.take_profit_1
            else:  # Short
                tp1_hit = current_price <= position.take_profit_1
            
            if tp1_hit:
                logger.success(
                    f"🎯 TAKE PROFIT 1 atingido: {symbol} @ {current_price:.2f} "
                    f"(TP1: {position.take_profit_1:.2f})"
                )
                
                # Fecha parte da posição
                qty_to_close = position.quantity * self.partial_tp1_pct
                self._close_partial_position(
                    position,
                    qty_to_close,
                    current_price,
                    "TP1"
                )
                
                # Marca TP1 como atingido
                position.tp1_hit = True
                self.risk_manager.mark_tp1_hit(symbol)
                
                # Move SL para breakeven
                if not position.sl_moved_to_breakeven:
                    self._move_sl_to_breakeven(position)
        
        # ====================================================================
        # VERIFICA TAKE PROFIT 2
        # ====================================================================
        
        tp2_hit = False
        
        if position.tp1_hit and position.quantity_remaining > 0:
            if position.side == "Long":
                tp2_hit = current_price >= position.take_profit_2
            else:  # Short
                tp2_hit = current_price <= position.take_profit_2
            
            if tp2_hit:
                logger.success(
                    f"🎯 TAKE PROFIT 2 atingido: {symbol} @ {current_price:.2f} "
                    f"(TP2: {position.take_profit_2:.2f})"
                )
                
                # Fecha posição restante
                self._close_position_at_market(position, "TP2")
        
        # ====================================================================
        # TRAILING STOP LOSS (opcional)
        # ====================================================================
        
        if self.enable_trailing_sl and position.tp1_hit:
            self._update_trailing_sl(position, current_price)
    
    # ========================================================================
    # FECHAMENTO DE POSIÇÕES
    # ========================================================================
    
    def _close_partial_position(
        self,
        position: ManagedPosition,
        quantity: float,
        exit_price: float,
        reason: str
    ):
        """Fecha parte da posição"""
        symbol = position.symbol
        
        # Determina lado da ordem de fechamento
        close_side = "Sell" if position.side == "Long" else "Buy"
        
        # Envia ordem de fechamento
        close_result = self.place_market_order(symbol, close_side, quantity)
        
        if not close_result or not close_result.get("success"):
            logger.error(f"❌ Falha ao fechar parcial: {symbol}")
            return
        
        # Atualiza quantidade restante
        position.quantity_remaining -= quantity
        
        # Registra fechamento no risk manager
        self.risk_manager.close_position(
            symbol=symbol,
            exit_price=exit_price,
            quantity=quantity,
            reason=reason
        )
        
        logger.success(
            f"💰 Fechamento parcial: {quantity:.6f} {symbol} @ {exit_price:.2f} "
            f"({reason}) | Restante: {position.quantity_remaining:.6f}"
        )
    
    def _close_position_at_market(
        self,
        position: ManagedPosition,
        reason: str
    ):
        """Fecha posição inteira a mercado"""
        symbol = position.symbol
        quantity = position.quantity_remaining
        
        if quantity <= 0:
            logger.warning(f"Posição já fechada: {symbol}")
            with self.lock:
                if symbol in self.managed_positions:
                    del self.managed_positions[symbol]
            return
        
        # Pega preço atual
        current_price = self._get_current_price(symbol)
        if not current_price:
            logger.error(f"Não foi possível obter preço de {symbol}")
            return
        
        # Determina lado da ordem de fechamento
        close_side = "Sell" if position.side == "Long" else "Buy"
        
        # Envia ordem de fechamento
        close_result = self.place_market_order(symbol, close_side, quantity)
        
        if not close_result or not close_result.get("success"):
            logger.error(f"❌ Falha ao fechar posição: {symbol}")
            return
        
        # Registra fechamento no risk manager
        self.risk_manager.close_position(
            symbol=symbol,
            exit_price=current_price,
            quantity=quantity,
            reason=reason
        )
        
        # Remove posição gerenciada
        with self.lock:
            if symbol in self.managed_positions:
                del self.managed_positions[symbol]
        
        logger.success(
            f"💰 Posição fechada: {quantity:.6f} {symbol} @ {current_price:.2f} "
            f"({reason})"
        )
    
    def close_position_manual(self, symbol: str) -> bool:
        """Fecha posição manualmente"""
        with self.lock:
            if symbol not in self.managed_positions:
                logger.warning(f"Posição não encontrada: {symbol}")
                return False
            
            position = self.managed_positions[symbol]
        
        self._close_position_at_market(position, "MANUAL")
        return True
    
    # ========================================================================
    # GERENCIAMENTO DE STOP LOSS
    # ========================================================================
    
    def _move_sl_to_breakeven(self, position: ManagedPosition):
        """Move stop loss para breakeven (preço de entrada)"""
        symbol = position.symbol
        old_sl = position.stop_loss
        new_sl = position.entry_price
        
        position.stop_loss = new_sl
        position.sl_moved_to_breakeven = True
        
        # Atualiza no risk manager
        self.risk_manager.move_sl_to_breakeven(symbol)
        
        logger.success(
            f"🔄 SL movido para breakeven: {symbol} | "
            f"{old_sl:.2f} → {new_sl:.2f}"
        )
    
    def _update_trailing_sl(self, position: ManagedPosition, current_price: float):
        """Atualiza trailing stop loss"""
        if position.side == "Long":
            # Para long, SL sobe junto com o preço
            new_sl = current_price * (1 - self.trailing_sl_pct)
            
            if new_sl > position.stop_loss:
                old_sl = position.stop_loss
                position.stop_loss = new_sl
                
                self.risk_manager.update_stop_loss(position.symbol, new_sl)
                
                logger.info(
                    f"🔄 Trailing SL atualizado: {position.symbol} | "
                    f"{old_sl:.2f} → {new_sl:.2f}"
                )
        
        else:  # Short
            # Para short, SL desce junto com o preço
            new_sl = current_price * (1 + self.trailing_sl_pct)
            
            if new_sl < position.stop_loss:
                old_sl = position.stop_loss
                position.stop_loss = new_sl
                
                self.risk_manager.update_stop_loss(position.symbol, new_sl)
                
                logger.info(
                    f"🔄 Trailing SL atualizado: {position.symbol} | "
                    f"{old_sl:.2f} → {new_sl:.2f}"
                )
    
    # ========================================================================
    # EXECUÇÃO DE ORDENS (PAPER VS REAL)
    # ========================================================================
    
    def _execute_paper_order(self, order: Order) -> Dict:
        """Executa ordem em paper mode (simulado)"""
        # Simula a ordem via BybitWebSocket paper trading
        result = self.ws_client.place_order_paper(
            symbol=order.symbol,
            side=order.side.value,
            qty=order.quantity,
            order_type=order.order_type.value,
            price=order.price
        )
        
        return result
    
    def _execute_real_order(self, order: Order) -> Dict:
        """Executa ordem REAL na exchange"""
        logger.warning("⚠️ Enviando ordem REAL para a exchange!")
        
        result = self.ws_client.place_order_real(
            symbol=order.symbol,
            side=order.side.value,
            qty=order.quantity,
            order_type=order.order_type.value,
            price=order.price
        )
        
        return result
    
    # ========================================================================
    # UTILITÁRIOS
    # ========================================================================
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Obtém preço atual do símbolo"""
        # Tenta pegar do cache de candles do WebSocket
        try:
            with self.ws_client.lock:
                for tf in ["1", "5", "15"]:
                    if symbol in self.ws_client.candles and tf in self.ws_client.candles[symbol]:
                        candles = self.ws_client.candles[symbol][tf]
                        if candles:
                            return candles[-1].close
        except:
            pass
        
        # Fallback: consulta ticker via REST
        ticker = self.ws_client.get_ticker(symbol)
        if ticker:
            return float(ticker.get("lastPrice", 0))
        
        return None
    
    def _generate_order_id(self) -> str:
        """Gera ID único para ordem"""
        order_id = f"ORD_{self.next_order_id:08d}"
        self.next_order_id += 1
        return order_id
    
    def get_active_positions(self) -> List[Dict]:
        """Retorna lista de posições ativas"""
        with self.lock:
            return [pos.to_dict() for pos in self.managed_positions.values()]
    
    def get_order_history(self, limit: int = 50) -> List[Dict]:
        """Retorna histórico de ordens"""
        with self.lock:
            recent_orders = self.order_history[-limit:] if limit else self.order_history
            return [order.to_dict() for order in recent_orders]
    
    def __repr__(self) -> str:
        return (
            f"OrderManager("
            f"paper_mode={self.paper_mode}, "
            f"active_positions={len(self.managed_positions)}, "
            f"monitoring={self.monitoring_active})"
        )


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Exemplo de uso do Order Manager
    """
    from exchange.bybit_ws import BybitWebSocket
    from risk.risk_manager import RiskManager
    
    # Configuração de log
    logger.add(
        "logs/order_manager.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )
    
    # Inicializa componentes (exemplo fictício)
    ws_client = BybitWebSocket(
        api_key="test_key",
        api_secret="test_secret",
        paper_mode=True
    )
    
    risk_manager = RiskManager(
        initial_capital=10000.0,
        risk_per_trade=0.01
    )
    
    # Inicializa Order Manager
    order_manager = OrderManager(
        ws_client=ws_client,
        risk_manager=risk_manager,
        paper_mode=True
    )
    
    # Inicia monitoramento
    order_manager.start_monitoring()
    
    # Exemplo: abre posição
    position = order_manager.open_position_with_targets(
        symbol="BTCUSDT",
        side="Long",
        quantity=0.01,
        stop_loss=41500.0,
        take_profit_1=42500.0,
        take_profit_2=43000.0
    )
    
    if position:
        logger.success(f"Posição aberta: {position.to_dict()}")
        
        # Simula espera
        time.sleep(5)
        
        # Lista posições ativas
        active = order_manager.get_active_positions()
        logger.info(f"Posições ativas: {active}")
    
    # Para monitoramento
    order_manager.stop_monitoring()