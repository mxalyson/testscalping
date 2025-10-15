"""
Bybit WebSocket & REST API Integration
======================================

Módulo responsável por:
- Conexão WebSocket para dados em tempo real (candles)
- API REST para operações de ordem e consultas
- Suporte a Paper Trading Mode (simulação de ordens)
- Reconexão automática em caso de falha
- Cache de dados históricos

Autor: Trading Bot Team
"""

import json
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np
from loguru import logger
from pybit.unified_trading import WebSocket, HTTP
import websocket
from dataclasses import dataclass, field


# ============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ============================================================================

TIMEFRAME_MAP = {
    "1": 1,      # 1 minuto
    "5": 5,      # 5 minutos
    "15": 15,    # 15 minutos
    "60": 60,    # 1 hora
    "120": 120,  # 2 horas
    "240": 240,  # 4 horas
    "D": 1440,   # 1 dia
}

BYBIT_WS_PUBLIC = {
    "mainnet": "wss://stream.bybit.com/v5/public/linear",
    "testnet": "wss://stream-testnet.bybit.com/v5/public/linear"
}

BYBIT_REST = {
    "mainnet": "https://api.bybit.com",
    "testnet": "https://api-testnet.bybit.com"
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Candle:
    """Representa um candle OHLCV"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


@dataclass
class PaperOrder:
    """Simula uma ordem em paper trading mode"""
    order_id: str
    symbol: str
    side: str  # "Buy" ou "Sell"
    order_type: str  # "Market" ou "Limit"
    qty: float
    price: Optional[float]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str = "New"  # New, Filled, Cancelled
    filled_price: Optional[float] = None
    filled_time: Optional[int] = None
    pnl: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'qty': self.qty,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'status': self.status,
            'filled_price': self.filled_price,
            'filled_time': self.filled_time,
            'pnl': self.pnl
        }


# ============================================================================
# CLASSE PRINCIPAL: BybitWebSocket
# ============================================================================

class BybitWebSocket:
    """
    Cliente WebSocket para Bybit com suporte a:
    - Streaming de candles em tempo real
    - Paper trading mode (simulação)
    - Reconexão automática
    - Cache de dados históricos
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        paper_mode: bool = True
    ):
        """
        Inicializa o cliente Bybit
        
        Args:
            api_key: API key da Bybit
            api_secret: API secret da Bybit
            testnet: Se True, usa testnet. Se False, mainnet
            paper_mode: Se True, simula ordens sem enviar à exchange
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.paper_mode = paper_mode
        
        # Endpoints
        self.ws_url = BYBIT_WS_PUBLIC["testnet" if testnet else "mainnet"]
        self.rest_url = BYBIT_REST["testnet" if testnet else "mainnet"]
        
        # Cliente REST API
        self.session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )
        
        # WebSocket
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.is_connected = False
        self.should_reconnect = True
        
        # Storage de candles: {symbol: {timeframe: deque([Candle])}}
        self.candles: Dict[str, Dict[str, deque]] = {}
        self.max_candles_per_tf = 500  # Mantém últimos 500 candles
        
        # Callbacks
        self.on_candle_callbacks: List[Callable] = []
        
        # Paper Trading
        self.paper_orders: Dict[str, PaperOrder] = {}
        self.paper_positions: Dict[str, Dict] = {}
        self.paper_balance = 10000.0  # USDT inicial (paper)
        self.next_order_id = 1
        
        # Lock para thread safety
        self.lock = threading.RLock()
        
        logger.info(
            f"BybitWebSocket inicializado - "
            f"Mode: {'TESTNET' if testnet else 'MAINNET'} | "
            f"Paper: {paper_mode}"
        )
    
    # ========================================================================
    # WEBSOCKET - CONEXÃO E GERENCIAMENTO
    # ========================================================================
    
    def connect(self):
        """Conecta ao WebSocket"""
        if self.is_connected:
            logger.warning("WebSocket já está conectado")
            return
        
        logger.info(f"Conectando WebSocket: {self.ws_url}")
        
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # Inicia thread do WebSocket
        self.ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self.ws_thread.start()
        
        # Aguarda conexão
        timeout = 10
        start = time.time()
        while not self.is_connected and (time.time() - start) < timeout:
            time.sleep(0.1)
        
        if not self.is_connected:
            raise ConnectionError("Falha ao conectar WebSocket")
        
        logger.success("WebSocket conectado com sucesso")
    
    def _run_ws(self):
        """Executa o WebSocket em thread separada"""
        while self.should_reconnect:
            try:
                self.ws.run_forever()
            except Exception as e:
                logger.error(f"Erro no WebSocket: {e}")
            
            if self.should_reconnect:
                logger.warning("Reconectando em 5 segundos...")
                time.sleep(5)
    
    def disconnect(self):
        """Desconecta o WebSocket"""
        logger.info("Desconectando WebSocket...")
        self.should_reconnect = False
        self.is_connected = False
        
        if self.ws:
            self.ws.close()
        
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5)
        
        logger.info("WebSocket desconectado")
    
    def _on_open(self, ws):
        """Callback quando WebSocket conecta"""
        self.is_connected = True
        logger.success("WebSocket aberto")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Callback quando WebSocket fecha"""
        self.is_connected = False
        logger.warning(
            f"WebSocket fechado - Code: {close_status_code}, "
            f"Msg: {close_msg}"
        )
    
    def _on_error(self, ws, error):
        """Callback de erro do WebSocket"""
        logger.error(f"Erro WebSocket: {error}")
    
    def _on_message(self, ws, message):
        """Callback quando recebe mensagem do WebSocket"""
        try:
            data = json.loads(message)
            
            # Ignora mensagens de ping/pong e success
            if data.get("success") or data.get("op") == "pong":
                return
            
            # Processa candles
            if data.get("topic", "").startswith("kline"):
                self._process_candle(data)
        
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {e}")
    
    # ========================================================================
    # WEBSOCKET - SUBSCRIÇÕES
    # ========================================================================
    
    def subscribe_candles(
        self,
        symbol: str,
        timeframes: List[str]
    ):
        """
        Subscreve candles em tempo real
        
        Args:
            symbol: Par de trading (ex: "BTCUSDT")
            timeframes: Lista de timeframes (ex: ["15", "60", "240"])
        
        Example:
            >>> ws.subscribe_candles("BTCUSDT", ["15", "60"])
        """
        if not self.is_connected:
            raise ConnectionError("WebSocket não está conectado")
        
        # Inicializa storage
        if symbol not in self.candles:
            self.candles[symbol] = {}
        
        for tf in timeframes:
            if tf not in TIMEFRAME_MAP:
                logger.warning(f"Timeframe inválido: {tf}")
                continue
            
            # Cria deque para armazenar candles
            if tf not in self.candles[symbol]:
                self.candles[symbol][tf] = deque(maxlen=self.max_candles_per_tf)
            
            # Monta tópico (ex: "kline.15.BTCUSDT")
            topic = f"kline.{tf}.{symbol}"
            
            # Envia subscrição
            subscribe_msg = {
                "op": "subscribe",
                "args": [topic]
            }
            
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscrito: {topic}")
            
            # Baixa dados históricos para popular cache
            self._fetch_historical_candles(symbol, tf, limit=200)
    
    def _process_candle(self, data: Dict):
        """Processa dados de candle recebidos via WebSocket"""
        try:
            topic = data.get("topic", "")
            candle_data = data.get("data", [])
            
            if not candle_data:
                return
            
            # Parse topic: "kline.15.BTCUSDT"
            parts = topic.split(".")
            if len(parts) != 3:
                return
            
            timeframe = parts[1]
            symbol = parts[2]
            
            # Extrai dados do candle
            for c in candle_data:
                candle = Candle(
                    timestamp=int(c["start"]),
                    open=float(c["open"]),
                    high=float(c["high"]),
                    low=float(c["low"]),
                    close=float(c["close"]),
                    volume=float(c["volume"])
                )
                
                # Armazena no cache
                with self.lock:
                    if symbol in self.candles and timeframe in self.candles[symbol]:
                        # Remove candle antigo com mesmo timestamp (atualização)
                        candles_deque = self.candles[symbol][timeframe]
                        if candles_deque and candles_deque[-1].timestamp == candle.timestamp:
                            candles_deque.pop()
                        
                        candles_deque.append(candle)
                
                # Chama callbacks
                for callback in self.on_candle_callbacks:
                    try:
                        callback(symbol, timeframe, candle)
                    except Exception as e:
                        logger.error(f"Erro em callback: {e}")
        
        except Exception as e:
            logger.error(f"Erro ao processar candle: {e}")
    
    def on_candle(self, callback: Callable):
        """
        Registra callback para ser chamado a cada novo candle
        
        Args:
            callback: Função com assinatura (symbol, timeframe, candle)
        
        Example:
            >>> def my_callback(symbol, tf, candle):
            >>>     print(f"{symbol} {tf}: Close = {candle.close}")
            >>> ws.on_candle(my_callback)
        """
        self.on_candle_callbacks.append(callback)
    
    # ========================================================================
    # REST API - DADOS HISTÓRICOS
    # ========================================================================
    
    def _fetch_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200
    ) -> List[Candle]:
        """
        Baixa candles históricos via REST API
        
        Args:
            symbol: Par de trading
            timeframe: Timeframe (ex: "15", "60")
            limit: Número de candles (max 200)
        
        Returns:
            Lista de objetos Candle
        """
        try:
            # Converte timeframe para minutos
            interval_minutes = TIMEFRAME_MAP.get(timeframe)
            if not interval_minutes:
                logger.error(f"Timeframe inválido: {timeframe}")
                return []
            
            # Calcula timestamp de início
            end_time = int(time.time() * 1000)
            start_time = end_time - (interval_minutes * 60 * 1000 * limit)
            
            # Faz request
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=timeframe,
                start=start_time,
                end=end_time,
                limit=limit
            )
            
            if response["retCode"] != 0:
                logger.error(f"Erro ao buscar candles: {response['retMsg']}")
                return []
            
            # Processa resposta
            candles = []
            for item in reversed(response["result"]["list"]):  # Bybit retorna desc
                candle = Candle(
                    timestamp=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5])
                )
                candles.append(candle)
            
            # Armazena no cache
            with self.lock:
                if symbol not in self.candles:
                    self.candles[symbol] = {}
                if timeframe not in self.candles[symbol]:
                    self.candles[symbol][timeframe] = deque(maxlen=self.max_candles_per_tf)
                
                for c in candles:
                    self.candles[symbol][timeframe].append(c)
            
            logger.debug(f"Baixados {len(candles)} candles: {symbol} {timeframe}")
            return candles
        
        except Exception as e:
            logger.error(f"Erro ao buscar histórico: {e}")
            return []
    
    def get_candles_df(
        self,
        symbol: str,
        timeframe: str,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retorna candles como DataFrame pandas
        
        Args:
            symbol: Par de trading
            timeframe: Timeframe
            limit: Número de candles (None = todos)
        
        Returns:
            DataFrame com colunas: timestamp, open, high, low, close, volume
        """
        with self.lock:
            if symbol not in self.candles or timeframe not in self.candles[symbol]:
                logger.warning(f"Sem dados para {symbol} {timeframe}")
                return pd.DataFrame()
            
            candles_deque = self.candles[symbol][timeframe]
            
            if limit:
                candles_list = list(candles_deque)[-limit:]
            else:
                candles_list = list(candles_deque)
            
            if not candles_list:
                return pd.DataFrame()
            
            df = pd.DataFrame([c.to_dict() for c in candles_list])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('datetime')
            
            return df
    
    # ========================================================================
    # PAPER TRADING - SIMULAÇÃO DE ORDENS
    # ========================================================================
    
    def place_order_paper(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Simula envio de ordem (Paper Trading Mode)
        
        Args:
            symbol: Par de trading
            side: "Buy" ou "Sell"
            qty: Quantidade
            order_type: "Market" ou "Limit"
            price: Preço (para Limit orders)
            stop_loss: Preço de stop loss
            take_profit: Preço de take profit
        
        Returns:
            Dict com detalhes da ordem simulada
        """
        if not self.paper_mode:
            raise ValueError("Método apenas para paper mode")
        
        # Gera order ID
        order_id = f"PAPER_{self.next_order_id:08d}"
        self.next_order_id += 1
        
        # Pega preço atual
        current_price = self._get_current_price_paper(symbol)
        if not current_price:
            return {"success": False, "error": "Preço não disponível"}
        
        # Cria ordem
        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            price=price if order_type == "Limit" else current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Se market order, preenche imediatamente
        if order_type == "Market":
            order.status = "Filled"
            order.filled_price = current_price
            order.filled_time = int(time.time() * 1000)
            
            # Atualiza posição
            self._update_position_paper(order)
        
        # Armazena ordem
        with self.lock:
            self.paper_orders[order_id] = order
        
        logger.info(
            f"[PAPER] Ordem criada: {order_id} | {side} {qty} {symbol} @ "
            f"{order.filled_price or price}"
        )
        
        return {
            "success": True,
            "order_id": order_id,
            "status": order.status,
            "details": order.to_dict()
        }
    
    def _get_current_price_paper(self, symbol: str) -> Optional[float]:
        """Retorna último preço disponível para o símbolo"""
        with self.lock:
            # Tenta pegar do menor timeframe disponível
            for tf in ["1", "5", "15"]:
                if symbol in self.candles and tf in self.candles[symbol]:
                    candles = self.candles[symbol][tf]
                    if candles:
                        return candles[-1].close
        return None
    
    def _update_position_paper(self, order: PaperOrder):
        """Atualiza posição em paper trading"""
        symbol = order.symbol
        
        with self.lock:
            if symbol not in self.paper_positions:
                self.paper_positions[symbol] = {
                    "qty": 0.0,
                    "avg_price": 0.0,
                    "side": None,
                    "unrealized_pnl": 0.0
                }
            
            position = self.paper_positions[symbol]
            
            # Calcula nova posição
            if order.side == "Buy":
                new_qty = position["qty"] + order.qty
            else:  # Sell
                new_qty = position["qty"] - order.qty
            
            # Atualiza preço médio
            if new_qty != 0:
                position["avg_price"] = (
                    (position["avg_price"] * position["qty"] + 
                     order.filled_price * order.qty) / new_qty
                )
            else:
                position["avg_price"] = 0.0
            
            position["qty"] = new_qty
            position["side"] = "Long" if new_qty > 0 else "Short" if new_qty < 0 else None
    
    def get_positions_paper(self) -> Dict[str, Dict]:
        """Retorna posições abertas (paper mode)"""
        with self.lock:
            return {
                symbol: pos.copy()
                for symbol, pos in self.paper_positions.items()
                if pos["qty"] != 0
            }
    
    def get_balance_paper(self) -> float:
        """Retorna saldo simulado (paper mode)"""
        return self.paper_balance
    
    # ========================================================================
    # REST API - ORDENS REAIS (para uso futuro)
    # ========================================================================
    
    def place_order_real(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Envia ordem REAL para a Bybit
        
        ⚠️ ATENÇÃO: Esta função envia ordens REAIS com dinheiro REAL!
        Só use se souber o que está fazendo.
        """
        if self.paper_mode:
            raise ValueError(
                "Bot está em PAPER MODE. "
                "Altere paper_mode=False para enviar ordens reais."
            )
        
        try:
            # Prepara parâmetros
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty)
            }
            
            if order_type == "Limit" and price:
                params["price"] = str(price)
            
            if stop_loss:
                params["stopLoss"] = str(stop_loss)
            
            if take_profit:
                params["takeProfit"] = str(take_profit)
            
            # Envia ordem
            response = self.session.place_order(**params)
            
            if response["retCode"] == 0:
                logger.success(f"Ordem enviada: {response['result']}")
                return {
                    "success": True,
                    "order_id": response["result"]["orderId"],
                    "details": response["result"]
                }
            else:
                logger.error(f"Erro ao enviar ordem: {response['retMsg']}")
                return {
                    "success": False,
                    "error": response["retMsg"]
                }
        
        except Exception as e:
            logger.error(f"Exceção ao enviar ordem: {e}")
            return {"success": False, "error": str(e)}
    
    def get_positions_real(self) -> List[Dict]:
        """Consulta posições reais na Bybit"""
        try:
            response = self.session.get_positions(
                category="linear",
                settleCoin="USDT"
            )
            
            if response["retCode"] == 0:
                return response["result"]["list"]
            else:
                logger.error(f"Erro ao consultar posições: {response['retMsg']}")
                return []
        
        except Exception as e:
            logger.error(f"Erro ao consultar posições: {e}")
            return []
    
    # ========================================================================
    # UTILIDADES
    # ========================================================================
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Retorna informações de ticker do símbolo"""
        try:
            response = self.session.get_tickers(
                category="linear",
                symbol=symbol
            )
            
            if response["retCode"] == 0 and response["result"]["list"]:
                return response["result"]["list"][0]
            return None
        
        except Exception as e:
            logger.error(f"Erro ao buscar ticker: {e}")
            return None
    
    def __repr__(self) -> str:
        return (
            f"BybitWebSocket("
            f"connected={self.is_connected}, "
            f"paper_mode={self.paper_mode}, "
            f"testnet={self.testnet})"
        )


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Exemplo de como usar o módulo
    """
    
    # Configuração de log
    logger.add(
        "logs/bybit_ws.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )
    
    # Credenciais (substitua pelas suas)
    API_KEY = "your_api_key"
    API_SECRET = "your_api_secret"
    
    # Inicializa cliente
    ws_client = BybitWebSocket(
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=True,  # Use testnet para testes
        paper_mode=True
    )
    
    # Callback para processar candles
    def on_new_candle(symbol, timeframe, candle):
        logger.info(
            f"[{symbol}] {timeframe}m - "
            f"Close: {candle.close:.2f} | Volume: {candle.volume:.2f}"
        )
    
    # Registra callback
    ws_client.on_candle(on_new_candle)
    
    # Conecta e subscreve
    ws_client.connect()
    ws_client.subscribe_candles("BTCUSDT", ["15", "60", "240"])
    
    # Simula ordem em paper mode
    order_result = ws_client.place_order_paper(
        symbol="BTCUSDT",
        side="Buy",
        qty=0.001,
        order_type="Market",
        stop_loss=40000.0,
        take_profit=45000.0
    )
    
    logger.info(f"Resultado da ordem: {order_result}")
    
    # Mantém rodando
    try:
        while True:
            time.sleep(1)
            
            # A cada 30 segundos, mostra DataFrame de candles
            if int(time.time()) % 30 == 0:
                df = ws_client.get_candles_df("BTCUSDT", "15", limit=10)
                if not df.empty:
                    logger.info(f"\n{df.tail()}")
    
    except KeyboardInterrupt:
        logger.info("Encerrando...")
        ws_client.disconnect()