"""
Bot de Trading Automatizado - Script Principal (CORRIGIDO)
==========================================================
"""

import os
import sys
import time
import signal
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

# Importa módulos do bot
from exchange.bybit_ws import BybitWebSocket, Candle
from strategy.strategy_core import ScalpingStrategy, SignalType, TradingSignal
from risk.risk_manager import RiskManager
from execution.order_manager import OrderManager
from market_data.news_sentiment import NewsSentimentFilter
from optimization.dynamic_params import DynamicParameterManager
from market_data.support_resistance import SupportResistanceDetector

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

load_dotenv()

# Configurações da API
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
TESTNET = os.getenv("TESTNET", "false").lower() == "true"
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() == "true"

# Configurações de Trading
TRADING_PAIRS = os.getenv("TRADING_PAIRS", "BTCUSDT,ETHUSDT").split(",")
TIMEFRAMES = os.getenv("TIMEFRAMES", "15,60,120,240").split(",")
PRIMARY_TIMEFRAME = TIMEFRAMES[0] if TIMEFRAMES else "15"
CONFIRMATION_TIMEFRAME = TIMEFRAMES[1] if len(TIMEFRAMES) > 1 else "60"

# Configurações de Risco
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "2"))
LEVERAGE = float(os.getenv("LEVERAGE", "5"))
MAX_DAILY_DRAWDOWN = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.05"))
CIRCUIT_BREAKER_ENABLED = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"

# Configurações da Estratégia
ATR_MULTIPLIER_SL = float(os.getenv("ATR_MULTIPLIER_SL", "1.0"))
ATR_MULTIPLIER_TP1 = float(os.getenv("ATR_MULTIPLIER_TP1", "1.0"))
ATR_MULTIPLIER_TP2 = float(os.getenv("ATR_MULTIPLIER_TP2", "2.0"))
MOVE_SL_TO_BREAKEVEN = os.getenv("MOVE_SL_TO_BREAKEVEN", "true").lower() == "true"
MIN_ATR_THRESHOLD = float(os.getenv("MIN_ATR_THRESHOLD", "0.0"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))  # REDUZIDO

# Configurações de Log
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/trading_bot.log")

# ============================================================================
# CLASSE PRINCIPAL: TradingBot
# ============================================================================

class TradingBot:
    """Bot de Trading Automatizado"""
    
    def __init__(self):
        """Inicializa o bot e todos os componentes"""
        
        # Configura logging PRIMEIRO
        self._setup_logging()
        
        logger.info("=" * 70)
        logger.info("🤖 INICIANDO BOT DE TRADING AUTOMATIZADO")
        logger.info("=" * 70)
        
        # Valida credenciais
        if not API_KEY or not API_SECRET:
            logger.critical("❌ API_KEY e API_SECRET não configurados no .env")
            sys.exit(1)
        
        # Modo
        mode_str = "TESTNET" if TESTNET else "MAINNET"
        paper_str = "PAPER TRADING" if PAPER_MODE else "REAL TRADING ⚠️"
        logger.warning(f"Modo: {mode_str} | {paper_str}")
        
        if not PAPER_MODE:
            logger.critical("⚠️  ATENÇÃO: BOT EM MODO REAL!")
            logger.critical("⚠️  Você tem 10 segundos para cancelar (Ctrl+C)")
            time.sleep(10)
        
        # Estado do bot
        self.running = False
        self.last_signal_time: Dict[str, int] = {}
        self.signal_cooldown = 300000  # 5 minutos em ms
        
        # Inicializa componentes
        logger.info("🔧 Inicializando componentes...")
        
        # 1. WebSocket Client
        self.ws_client = BybitWebSocket(
            api_key=API_KEY,
            api_secret=API_SECRET,
            testnet=TESTNET,
            paper_mode=PAPER_MODE
        )
        
        # 2. Strategy
        self.strategy = ScalpingStrategy(
            atr_multiplier_sl=ATR_MULTIPLIER_SL,
            atr_multiplier_tp1=ATR_MULTIPLIER_TP1,
            atr_multiplier_tp2=ATR_MULTIPLIER_TP2,
            min_atr_threshold=MIN_ATR_THRESHOLD,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        # 3. Risk Manager
        self.risk_manager = RiskManager(
            initial_capital=INITIAL_CAPITAL,
            risk_per_trade=RISK_PER_TRADE,
            max_positions=MAX_POSITIONS,
            leverage=LEVERAGE,
            max_daily_drawdown=MAX_DAILY_DRAWDOWN,
            circuit_breaker_enabled=CIRCUIT_BREAKER_ENABLED
        )
        
        # 4. Order Manager
        self.order_manager = OrderManager(
            ws_client=self.ws_client,
            risk_manager=self.risk_manager,
            paper_mode=PAPER_MODE
        )
        
        # 5. News Filter (opcional)
        ENABLE_NEWS_FILTER = os.getenv("ENABLE_NEWS_FILTER", "false").lower() == "true"
        if ENABLE_NEWS_FILTER:
            self.news_filter = NewsSentimentFilter(
                enable_sentiment_analysis=True,
                sentiment_engine="keywords"
            )
            logger.info("✅ News Filter habilitado")
        else:
            self.news_filter = None
            logger.info("⚠️  News Filter desabilitado")
        
        # 6. Dynamic Parameters (opcional)
        ENABLE_DYNAMIC_PARAMS = os.getenv("ENABLE_DYNAMIC_PARAMS", "false").lower() == "true"
        if ENABLE_DYNAMIC_PARAMS:
            self.dynamic_params = DynamicParameterManager(
                base_atr_multiplier_sl=ATR_MULTIPLIER_SL,
                base_atr_multiplier_tp1=ATR_MULTIPLIER_TP1,
                base_atr_multiplier_tp2=ATR_MULTIPLIER_TP2,
                base_risk_per_trade=RISK_PER_TRADE,
                enable_volatility_adjustment=True,
                enable_trend_adjustment=True
            )
            logger.info("✅ Dynamic Parameters habilitado")
        else:
            self.dynamic_params = None
            logger.info("⚠️  Dynamic Parameters desabilitado")
        
        # 7. Support/Resistance Detector (opcional)
        ENABLE_SR_DETECTION = os.getenv("ENABLE_SR_DETECTION", "false").lower() == "true"
        if ENABLE_SR_DETECTION:
            self.sr_detector = SupportResistanceDetector(
                swing_window=int(os.getenv("SR_SWING_WINDOW", "5")),
                cluster_threshold=float(os.getenv("SR_CLUSTER_THRESHOLD", "0.002")),
                min_touches=int(os.getenv("SR_MIN_TOUCHES", "2")),
                min_strength=float(os.getenv("SR_MIN_STRENGTH", "0.3")),
                zone_width_pct=float(os.getenv("SR_ZONE_WIDTH", "0.003")),
                lookback_periods=int(os.getenv("SR_LOOKBACK", "200")),
                recency_weight=0.3,
                volume_weight=0.3
            )
            logger.info("✅ Support/Resistance Detection habilitado")
        else:
            self.sr_detector = None
            logger.info("⚠️  Support/Resistance Detection desabilitado")
        
        logger.success("✅ Componentes inicializados")
        
        # Registra handlers de sinal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._print_config()
    
    def _setup_logging(self):
        """Configura sistema de logging"""
        # Remove handlers padrão
        logger.remove()
        
        # Console output
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level=LOG_LEVEL,
            colorize=True
        )
        
        # Arquivo de log
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        logger.add(
            LOG_FILE,
            rotation="1 day",
            retention="30 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        )
    
    def _print_config(self):
        """Imprime configurações do bot"""
        logger.info("📋 Configurações:")
        logger.info(f"   Pares: {', '.join(TRADING_PAIRS)}")
        logger.info(f"   Timeframes: {', '.join(TIMEFRAMES)}m")
        logger.info(f"   Capital Inicial: {INITIAL_CAPITAL:.2f} USDT")
        logger.info(f"   Risco por Trade: {RISK_PER_TRADE*100:.1f}%")
        logger.info(f"   Max Posições: {MAX_POSITIONS}")
        logger.info(f"   Alavancagem: {LEVERAGE}x")
        logger.info(f"   SL: {ATR_MULTIPLIER_SL}×ATR | TP1: {ATR_MULTIPLIER_TP1}×ATR | TP2: {ATR_MULTIPLIER_TP2}×ATR")
        logger.info(f"   Confiança Mínima: {CONFIDENCE_THRESHOLD*100:.0f}%")
        logger.info(f"   Circuit Breaker: {'Ativado' if CIRCUIT_BREAKER_ENABLED else 'Desativado'}")
    
    def _signal_handler(self, signum, frame):
        """Handler para shutdown gracioso"""
        logger.warning(f"\n⚠️  Recebido sinal de interrupção ({signum})")
        self.stop()
    
    # ========================================================================
    # MÉTODOS PRINCIPAIS
    # ========================================================================
    
    def start(self):
        """Inicia o bot"""
        logger.info("🚀 Iniciando bot...")
        
        try:
            # Conecta WebSocket
            logger.info("📌 Conectando ao WebSocket...")
            self.ws_client.connect()
            
            # Subscreve candles
            logger.info(f"📡 Subscrevendo candles para {len(TRADING_PAIRS)} pares...")
            for symbol in TRADING_PAIRS:
                self.ws_client.subscribe_candles(symbol, TIMEFRAMES)
            
            # Aguarda dados iniciais
            logger.info("⏳ Aguardando dados iniciais (5 segundos)...")
            time.sleep(5)
            
            # Inicia monitoramento de posições
            logger.info("🔍 Iniciando monitoramento de posições...")
            self.order_manager.start_monitoring()
            
            # Marca como rodando
            self.running = True
            
            logger.success("✅ Bot iniciado com sucesso!")
            logger.info("=" * 70)
            logger.info("🤖 BOT RODANDO - Aguardando sinais...")
            logger.info("=" * 70)
            
            # Loop principal
            self._main_loop()
        
        except KeyboardInterrupt:
            logger.warning("⚠️  Interrompido pelo usuário")
            self.stop()
        
        except Exception as e:
            logger.critical(f"❌ Erro crítico: {e}")
            logger.exception(e)
            self.stop()
    
    def _main_loop(self):
        """Loop principal do bot"""
        check_interval = 15  # Verifica a cada 15 segundos
        last_status_print = time.time()
        status_interval = 300  # Imprime status a cada 5 minutos
        
        while self.running:
            try:
                current_time = time.time()
                
                # Imprime status periodicamente
                if current_time - last_status_print >= status_interval:
                    self._print_status()
                    last_status_print = current_time
                
                # Verifica circuit breaker
                if self.risk_manager.circuit_breaker_active:
                    if not hasattr(self, '_cb_warning_shown'):
                        logger.warning("🚨 Circuit breaker ativo - aguardando reset")
                        self._cb_warning_shown = True
                    time.sleep(check_interval)
                    continue
                else:
                    if hasattr(self, '_cb_warning_shown'):
                        delattr(self, '_cb_warning_shown')
                
                # Analisa cada par de trading
                for symbol in TRADING_PAIRS:
                    self._analyze_symbol(symbol)
                
                # Aguarda próximo ciclo
                time.sleep(check_interval)
            
            except Exception as e:
                logger.error(f"Erro no loop principal: {e}")
                time.sleep(30)
    
    def _analyze_symbol(self, symbol: str):
        """
        Analisa um símbolo e gera sinais
        
        Args:
            symbol: Símbolo a analisar
        """
        try:
            # Verifica se já tem posição aberta neste símbolo
            if symbol in self.order_manager.managed_positions:
                return
            
            # Verifica cooldown de sinal
            if symbol in self.last_signal_time:
                time_since_last = int(time.time() * 1000) - self.last_signal_time[symbol]
                if time_since_last < self.signal_cooldown:
                    return
            
            # Pega dados do timeframe primário
            df_primary = self.ws_client.get_candles_df(symbol, PRIMARY_TIMEFRAME)
            
            if df_primary.empty or len(df_primary) < 100:
                return
            
            # Analisa e gera sinal
            signal = self.strategy.analyze(df_primary, symbol, PRIMARY_TIMEFRAME)
            
            if not signal:
                return
            
            # ================================================================
            # CONFIRMAÇÃO MULTI-TIMEFRAME
            # ================================================================
            
            logger.info(f"🔍 Sinal detectado: {signal.signal_type.value} {symbol} (confiança: {signal.confidence:.1%})")
            logger.info(f"   Confirmando com timeframe {CONFIRMATION_TIMEFRAME}m...")
            
            # Pega dados do timeframe de confirmação
            df_confirmation = self.ws_client.get_candles_df(symbol, CONFIRMATION_TIMEFRAME)
            
            if df_confirmation.empty or len(df_confirmation) < 50:
                logger.warning(f"   ❌ Dados insuficientes no TF {CONFIRMATION_TIMEFRAME}m")
                return
            
            # Confirma sinal
            confirmed = self.strategy.confirm_signal_multi_tf(
                signal_primary=signal,
                df_higher_tf=df_confirmation,
                higher_tf_name=f"{CONFIRMATION_TIMEFRAME}m"
            )
            
            if not confirmed:
                logger.warning(f"   ❌ Sinal não confirmado pelo TF superior")
                return
            
            logger.success(f"   ✅ Sinal confirmado!")
            
            # ================================================================
            # AJUSTE DINÂMICO DE PARÂMETROS (se habilitado)
            # ================================================================
            
            if self.dynamic_params:
                logger.info(f"   🔧 Ajustando parâmetros dinamicamente...")
                
                df_for_analysis = self.ws_client.get_candles_df(symbol, PRIMARY_TIMEFRAME, limit=200)
                adjusted_params = self.dynamic_params.adjust_parameters(df=df_for_analysis)
                
                # Recalcula SL/TP com parâmetros ajustados
                entry_price = signal.entry_price
                atr = df_for_analysis.iloc[-1]['atr']
                
                if signal.signal_type == SignalType.LONG:
                    signal.stop_loss = entry_price - (atr * adjusted_params.atr_multiplier_sl)
                    signal.take_profit_1 = entry_price + (atr * adjusted_params.atr_multiplier_tp1)
                    signal.take_profit_2 = entry_price + (atr * adjusted_params.atr_multiplier_tp2)
                else:  # SHORT
                    signal.stop_loss = entry_price + (atr * adjusted_params.atr_multiplier_sl)
                    signal.take_profit_1 = entry_price - (atr * adjusted_params.atr_multiplier_tp1)
                    signal.take_profit_2 = entry_price - (atr * adjusted_params.atr_multiplier_tp2)
                
                # Atualiza risk manager
                self.risk_manager.risk_per_trade = adjusted_params.risk_per_trade
                
                logger.info(f"      SL: {signal.stop_loss:.2f} (ATR×{adjusted_params.atr_multiplier_sl:.2f})")
                logger.info(f"      TP1: {signal.take_profit_1:.2f} (ATR×{adjusted_params.atr_multiplier_tp1:.2f})")
                logger.info(f"      TP2: {signal.take_profit_2:.2f} (ATR×{adjusted_params.atr_multiplier_tp2:.2f})")
            
            # ================================================================
            # ANÁLISE DE SUPPORT/RESISTANCE (se habilitado)
            # ================================================================
            
            if self.sr_detector:
                logger.info(f"   📍 Analisando níveis de S/R...")
                
                df_for_sr = self.ws_client.get_candles_df(symbol, PRIMARY_TIMEFRAME, limit=200)
                sr_analysis = self.sr_detector.detect(
                    df=df_for_sr,
                    current_price=signal.entry_price,
                    methods=["swing", "cluster"]
                )
                
                logger.info(f"      Detectados: {len(sr_analysis.support_levels)} suportes, {len(sr_analysis.resistance_levels)} resistências")
                
                # Validação com S/R
                ENABLE_SR_FILTER = os.getenv("ENABLE_SR_FILTER", "true").lower() == "true"
                
                if ENABLE_SR_FILTER:
                    if (signal.signal_type == SignalType.LONG and 
                        sr_analysis.price_position == "near_resistance" and
                        sr_analysis.nearest_resistance and
                        sr_analysis.nearest_resistance.strength > 0.7):
                        
                        logger.warning(f"   ❌ LONG bloqueado: muito próximo de resistência forte")
                        return
                    
                    if (signal.signal_type == SignalType.SHORT and 
                        sr_analysis.price_position == "near_support" and
                        sr_analysis.nearest_support and
                        sr_analysis.nearest_support.strength > 0.7):
                        
                        logger.warning(f"   ❌ SHORT bloqueado: muito próximo de suporte forte")
                        return
            
            # ================================================================
            # VERIFICAÇÃO DE NOTÍCIAS (se habilitado)
            # ================================================================
            
            if self.news_filter:
                logger.info(f"   📰 Verificando notícias recentes...")
                news_result = self.news_filter.should_trade(
                    symbol=symbol,
                    hours_back=2,
                    min_sentiment_score=-0.3
                )
                
                if not news_result.should_trade:
                    logger.warning(f"   ❌ Trade bloqueado por notícias: {news_result.reason}")
                    return
                
                logger.success(f"   ✅ Notícias OK (sentiment: {news_result.sentiment_score:.2f})")
            
            # ================================================================
            # CÁLCULO DE POSITION SIZE
            # ================================================================
            
            quantity, sizing_details = self.risk_manager.calculate_position_size(
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                symbol=symbol,
                signal_type=signal.signal_type.value
            )
            
            if sizing_details.get("status") != "APPROVED":
                logger.warning(f"   ❌ Position sizing rejeitado: {sizing_details.get('error')}")
                return
            
            # ================================================================
            # EXECUÇÃO DO TRADE
            # ================================================================
            
            logger.info(f"🚀 Executando trade: {signal.signal_type.value} {symbol}")
            
            # Determina lado
            side = "Long" if signal.signal_type == SignalType.LONG else "Short"
            
            # Abre posição com SL/TP
            position = self.order_manager.open_position_with_targets(
                symbol=symbol,
                side=side,
                quantity=quantity,
                stop_loss=signal.stop_loss,
                take_profit_1=signal.take_profit_1,
                take_profit_2=signal.take_profit_2
            )
            
            if position:
                logger.success(
                    f"✅ TRADE ABERTO: {side} {quantity:.6f} {symbol}\n"
                    f"   Entry: {signal.entry_price:.2f}\n"
                    f"   SL: {signal.stop_loss:.2f}\n"
                    f"   TP1: {signal.take_profit_1:.2f} (fecha 50%)\n"
                    f"   TP2: {signal.take_profit_2:.2f} (fecha restante)\n"
                    f"   Razão: {signal.reason}"
                )
                
                # Atualiza tempo do último sinal
                self.last_signal_time[symbol] = int(time.time() * 1000)
            else:
                logger.error(f"   ❌ Falha ao abrir posição")
        
        except Exception as e:
            logger.error(f"Erro ao analisar {symbol}: {e}")
            logger.exception(e)
    
    def _print_status(self):
        """Imprime status atual do bot"""
        logger.info("=" * 70)
        logger.info("📊 STATUS DO BOT")
        logger.info("=" * 70)
        
        status = self.risk_manager.get_risk_status()
        
        logger.info(f"💰 Capital: {status['current_capital']:.2f} USDT")
        logger.info(
            f"📈 PnL Total: {status['total_pnl']:+.2f} USDT "
            f"({status['total_pnl_pct']:+.2f}%)"
        )
        logger.info(
            f"📊 PnL Diário: {status['daily_pnl']:+.2f} USDT "
            f"(DD: {status['daily_drawdown_pct']:.2f}%)"
        )
        logger.info(
            f"🎯 Win Rate: {status['win_rate']:.1f}% "
            f"({status['winning_trades']}W / {status['losing_trades']}L)"
        )
        logger.info(
            f"📦 Posições: {status['open_positions']}/{status['max_positions']}"
        )
        logger.info(f"💸 Fees Totais: {status['total_fees']:.2f} USDT")
        
        positions = self.order_manager.get_active_positions()
        if positions:
            logger.info(f"\n📍 Posições Abertas ({len(positions)}):")
            for pos in positions:
                logger.info(
                    f"   • {pos['symbol']}: {pos['side']} "
                    f"{pos['quantity_remaining']:.6f} @ {pos['entry_price']:.2f}"
                )
        else:
            logger.info("\n📍 Nenhuma posição aberta")
        
        logger.info("=" * 70)
    
    def stop(self):
        """Para o bot de forma controlada"""
        if not self.running:
            return
        
        logger.warning("🛑 Parando bot...")
        self.running = False
        
        try:
            logger.info("   Parando monitoramento de posições...")
            self.order_manager.stop_monitoring()
            
            logger.info("   Desconectando WebSocket...")
            self.ws_client.disconnect()
            
            self._print_final_summary()
            
            logger.success("✅ Bot parado com sucesso")
        
        except Exception as e:
            logger.error(f"Erro ao parar bot: {e}")
    
    def _print_final_summary(self):
        """Imprime resumo final antes de encerrar"""
        logger.info("=" * 70)
        logger.info("📊 RESUMO FINAL")
        logger.info("=" * 70)
        
        status = self.risk_manager.get_risk_status()
        
        logger.info(f"💰 Capital Final: {status['current_capital']:.2f} USDT")
        logger.info(
            f"📈 PnL Total: {status['total_pnl']:+.2f} USDT "
            f"({status['total_pnl_pct']:+.2f}%)"
        )
        logger.info(f"🎯 Total de Trades: {status['total_trades']}")
        logger.info(
            f"✅ Wins: {status['winning_trades']} | "
            f"❌ Losses: {status['losing_trades']} | "
            f"Win Rate: {status['win_rate']:.1f}%"
        )
        logger.info(f"💸 Fees Pagos: {status['total_fees']:.2f} USDT")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/trade_history_{timestamp}.csv"
            self.risk_manager.export_trade_history(filename)
            logger.info(f"📁 Histórico exportado: {filename}")
        except Exception as e:
            logger.error(f"Erro ao exportar histórico: {e}")
        
        logger.info("=" * 70)


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal"""
    
    print("\n" + "=" * 70)
    print("🤖  BOT DE TRADING AUTOMATIZADO - BYBIT FUTURES SCALPING")
    print("=" * 70)
    print("⚠️   USO POR SUA CONTA E RISCO")
    print("⚠️   SEMPRE TESTE EM PAPER MODE ANTES DE USAR CAPITAL REAL")
    print("=" * 70 + "\n")
    
    if not os.path.exists(".env"):
        logger.critical("❌ Arquivo .env não encontrado!")
        logger.info("   Copie .env.example para .env e configure suas credenciais")
        sys.exit(1)
    
    bot = TradingBot()
    
    try:
        bot.start()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrompido pelo usuário")
        bot.stop()
    except Exception as e:
        logger.critical(f"❌ Erro fatal: {e}")
        logger.exception(e)
        bot.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
