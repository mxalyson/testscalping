"""
Dashboard Streamlit - Interface Visual do Bot
=============================================

Dashboard interativo para:
- Visualizar performance em tempo real
- Controlar o bot (iniciar/parar)
- Ajustar parâmetros
- Ver gráficos de candles com sinais
- Exportar relatórios
- Rodar backtests

Uso:
    streamlit run ui/dashboard.py

Autor: Trading Bot Team
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import sys
from pathlib import Path

# Adiciona diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from exchange.bybit_ws import BybitWebSocket
from strategy.strategy_core import ScalpingStrategy
from risk.risk_manager import RiskManager
from execution.order_manager import OrderManager
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()


# ============================================================================
# CONFIGURAÇÕES DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .profit {
        color: #00c853;
        font-weight: bold;
    }
    .loss {
        color: #d32f2f;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# INICIALIZAÇÃO DO ESTADO DA SESSÃO
# ============================================================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.bot_running = False
    st.session_state.ws_client = None
    st.session_state.risk_manager = None
    st.session_state.order_manager = None
    st.session_state.strategy = None
    st.session_state.last_update = time.time()


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def initialize_components():
    """Inicializa todos os componentes do bot"""
    try:
        # Credenciais
        api_key = os.getenv("API_KEY", "")
        api_secret = os.getenv("API_SECRET", "")
        testnet = os.getenv("TESTNET", "false").lower() == "true"
        paper_mode = os.getenv("PAPER_MODE", "true").lower() == "true"
        
        if not api_key or not api_secret:
            st.error("❌ API_KEY e API_SECRET não configurados no .env")
            return False
        
        # WebSocket
        st.session_state.ws_client = BybitWebSocket(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            paper_mode=paper_mode
        )
        
        # Strategy
        st.session_state.strategy = ScalpingStrategy(
            atr_multiplier_sl=float(os.getenv("ATR_MULTIPLIER_SL", "1.0")),
            atr_multiplier_tp1=float(os.getenv("ATR_MULTIPLIER_TP1", "1.0")),
            atr_multiplier_tp2=float(os.getenv("ATR_MULTIPLIER_TP2", "2.0")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
        )
        
        # Risk Manager
        st.session_state.risk_manager = RiskManager(
            initial_capital=float(os.getenv("INITIAL_CAPITAL", "10000")),
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.01")),
            max_positions=int(os.getenv("MAX_POSITIONS", "2")),
            leverage=float(os.getenv("LEVERAGE", "5")),
            max_daily_drawdown=float(os.getenv("MAX_DAILY_DRAWDOWN", "0.05"))
        )
        
        # Order Manager
        st.session_state.order_manager = OrderManager(
            ws_client=st.session_state.ws_client,
            risk_manager=st.session_state.risk_manager,
            paper_mode=paper_mode
        )
        
        st.session_state.initialized = True
        return True
    
    except Exception as e:
        st.error(f"❌ Erro ao inicializar componentes: {e}")
        return False


def start_bot():
    """Inicia o bot"""
    try:
        if not st.session_state.initialized:
            if not initialize_components():
                return False
        
        # Conecta WebSocket
        st.session_state.ws_client.connect()
        
        # Subscreve candles
        trading_pairs = os.getenv("TRADING_PAIRS", "BTCUSDT").split(",")
        timeframes = os.getenv("TIMEFRAMES", "15,60").split(",")
        
        for symbol in trading_pairs:
            st.session_state.ws_client.subscribe_candles(symbol, timeframes)
        
        # Inicia monitoramento
        st.session_state.order_manager.start_monitoring()
        
        st.session_state.bot_running = True
        st.success("✅ Bot iniciado com sucesso!")
        return True
    
    except Exception as e:
        st.error(f"❌ Erro ao iniciar bot: {e}")
        return False


def stop_bot():
    """Para o bot"""
    try:
        if st.session_state.order_manager:
            st.session_state.order_manager.stop_monitoring()
        
        if st.session_state.ws_client:
            st.session_state.ws_client.disconnect()
        
        st.session_state.bot_running = False
        st.success("✅ Bot parado com sucesso!")
        return True
    
    except Exception as e:
        st.error(f"❌ Erro ao parar bot: {e}")
        return False


def create_candlestick_chart(df: pd.DataFrame, symbol: str, signals: Optional[List] = None):
    """
    Cria gráfico de candles com indicadores
    
    Args:
        df: DataFrame com dados OHLCV e indicadores
        symbol: Símbolo do ativo
        signals: Lista de sinais para marcar no gráfico
    """
    # Cria subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{symbol} - Candlestick", "RSI", "MACD")
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#00c853',
            decreasing_line_color='#d32f2f'
        ),
        row=1, col=1
    )
    
    # EMAs
    if 'ema_fast' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema_fast'],
                name='EMA 9',
                line=dict(color='#ff9800', width=1)
            ),
            row=1, col=1
        )
    
    if 'ema_medium' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema_medium'],
                name='EMA 21',
                line=dict(color='#2196f3', width=1)
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'bb_upper' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_lower'],
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                opacity=0.3
            ),
            row=1, col=1
        )
    
    # Sinais de entrada/saída
    if signals:
        for signal in signals:
            color = '#00c853' if signal['type'] == 'LONG' else '#d32f2f'
            symbol_marker = 'triangle-up' if signal['type'] == 'LONG' else 'triangle-down'
            
            fig.add_trace(
                go.Scatter(
                    x=[signal['time']],
                    y=[signal['price']],
                    mode='markers',
                    marker=dict(
                        symbol=symbol_marker,
                        size=15,
                        color=color,
                        line=dict(width=2, color='white')
                    ),
                    name=f"{signal['type']} Entry",
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                name='RSI',
                line=dict(color='#9c27b0', width=2)
            ),
            row=2, col=1
        )
        
        # Linhas de referência RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
    
    # MACD
    if 'macd' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd'],
                name='MACD',
                line=dict(color='#2196f3', width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd_signal'],
                name='Signal',
                line=dict(color='#ff9800', width=2)
            ),
            row=3, col=1
        )
        
        # Histograma MACD
        if 'macd_diff' in df.columns:
            colors = ['#00c853' if val >= 0 else '#d32f2f' for val in df['macd_diff']]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['macd_diff'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.5
                ),
                row=3, col=1
            )
    
    # Layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig


# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

def main():
    """Interface principal do dashboard"""
    
    # Header
    st.markdown('<p class="main-header">🤖 Trading Bot Dashboard</p>', unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR - CONTROLES
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Controles")
        
        # Status do bot
        if st.session_state.bot_running:
            st.success("🟢 Bot Ativo")
        else:
            st.error("🔴 Bot Inativo")
        
        st.markdown("---")
        
        # Botões de controle
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Iniciar Bot", disabled=st.session_state.bot_running):
                start_bot()
                st.rerun()
        
        with col2:
            if st.button("⏹️ Parar Bot", disabled=not st.session_state.bot_running):
                stop_bot()
                st.rerun()
        
        st.markdown("---")
        
        # Configurações
        st.subheader("📋 Configurações")
        
        # Pares de trading
        trading_pairs_input = st.text_input(
            "Pares de Trading",
            value=os.getenv("TRADING_PAIRS", "BTCUSDT,ETHUSDT"),
            help="Separe por vírgula"
        )
        
        # Timeframes
        primary_tf = st.selectbox(
            "Timeframe Primário",
            options=["1", "5", "15", "30", "60"],
            index=2,
            help="Timeframe para análise principal"
        )
        
        confirmation_tf = st.selectbox(
            "Timeframe Confirmação",
            options=["15", "30", "60", "120", "240"],
            index=2,
            help="Timeframe para confirmar sinais"
        )
        
        # Risk settings
        st.subheader("🛡️ Gestão de Risco")
        
        risk_per_trade = st.slider(
            "Risco por Trade (%)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Percentual do capital arriscado por trade"
        )
        
        max_positions = st.number_input(
            "Max Posições Simultâneas",
            min_value=1,
            max_value=10,
            value=2,
            help="Número máximo de posições abertas"
        )
        
        leverage = st.slider(
            "Alavancagem",
            min_value=1,
            max_value=20,
            value=5,
            help="Alavancagem para trades"
        )
        
        st.markdown("---")
        
        # Auto-refresh
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
        if auto_refresh:
            refresh_interval = st.slider(
                "Intervalo (segundos)",
                min_value=1,
                max_value=30,
                value=5
            )
    
    # ========================================================================
    # TABS PRINCIPAIS
    # ========================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "💹 Gráficos",
        "📈 Posições",
        "📜 Histórico",
        "⚙️ Backtest"
    ])
    
    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    
    with tab1:
        if not st.session_state.initialized:
            st.warning("⚠️ Componentes não inicializados. Clique em 'Iniciar Bot' na sidebar.")
        else:
            # Métricas principais
            st.subheader("💰 Performance")
            
            status = st.session_state.risk_manager.get_risk_status()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Capital Atual",
                    f"${status['current_capital']:.2f}",
                    delta=f"{status['total_pnl_pct']:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "PnL Total",
                    f"${status['total_pnl']:+.2f}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "PnL Diário",
                    f"${status['daily_pnl']:+.2f}",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Win Rate",
                    f"{status['win_rate']:.1f}%",
                    delta=None
                )
            
            # Segunda linha de métricas
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric(
                    "Posições Abertas",
                    f"{status['open_positions']}/{status['max_positions']}"
                )
            
            with col6:
                st.metric(
                    "Total Trades",
                    status['total_trades']
                )
            
            with col7:
                st.metric(
                    "Trades Vencedores",
                    status['winning_trades']
                )
            
            with col8:
                st.metric(
                    "Fees Totais",
                    f"${status['total_fees']:.2f}"
                )
            
            # Status do Circuit Breaker
            if status['circuit_breaker_active']:
                st.error("🚨 CIRCUIT BREAKER ATIVO - Trading pausado")
            
            # Drawdown
            st.subheader("📉 Drawdown")
            
            col9, col10 = st.columns(2)
            
            with col9:
                dd_daily = status['daily_drawdown_pct']
                dd_max = status['max_daily_drawdown_pct']
                progress_daily = min(dd_daily / dd_max, 1.0) if dd_max > 0 else 0
                
                st.progress(progress_daily)
                st.caption(f"Drawdown Diário: {dd_daily:.2f}% / {dd_max:.1f}%")
            
            with col10:
                dd_total = status['total_drawdown_pct']
                dd_total_max = status['max_total_drawdown_pct']
                progress_total = min(dd_total / dd_total_max, 1.0) if dd_total_max > 0 else 0
                
                st.progress(progress_total)
                st.caption(f"Drawdown Total: {dd_total:.2f}% / {dd_total_max:.1f}%")
            
            # Métricas de Performance
            if status['total_trades'] > 0:
                st.subheader("📈 Métricas Avançadas")
                
                metrics = st.session_state.risk_manager.get_performance_metrics()
                
                col11, col12, col13, col14 = st.columns(4)
                
                with col11:
                    st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                
                with col12:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                
                with col13:
                    st.metric("Avg Win", f"${metrics['avg_win']:.2f}")
                
                with col14:
                    st.metric("Avg Loss", f"${metrics['avg_loss']:.2f}")
    
    # ========================================================================
    # TAB 2: GRÁFICOS
    # ========================================================================
    
    with tab2:
        if not st.session_state.bot_running:
            st.warning("⚠️ Inicie o bot para visualizar gráficos em tempo real")
        else:
            st.subheader("📊 Análise Técnica")
            
            # Seleção de símbolo
            trading_pairs = os.getenv("TRADING_PAIRS", "BTCUSDT").split(",")
            selected_symbol = st.selectbox("Selecione o Par", trading_pairs)
            
            # Seleção de timeframe
            selected_tf = st.selectbox(
                "Timeframe",
                options=["15", "60", "120", "240"],
                index=0
            )
            
            # Pega dados
            df = st.session_state.ws_client.get_candles_df(
                selected_symbol,
                selected_tf,
                limit=100
            )
            
            if not df.empty:
                # Calcula indicadores
                df = st.session_state.strategy.calculate_indicators(df)
                
                # Cria gráfico
                fig = create_candlestick_chart(df, selected_symbol)
                st.plotly_chart(fig, use_container_width=True)
                
                # Últimos valores
                st.subheader("📍 Valores Atuais")
                
                last_candle = df.iloc[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Preço", f"${last_candle['close']:.2f}")
                
                with col2:
                    st.metric("RSI", f"{last_candle['rsi']:.1f}")
                
                with col3:
                    st.metric("ATR", f"{last_candle['atr']:.2f}")
                
                with col4:
                    ema_trend = "Bullish" if last_candle['ema_fast'] > last_candle['ema_medium'] else "Bearish"
                    st.metric("Tendência EMA", ema_trend)
            else:
                st.info("Aguardando dados...")
    
    # ========================================================================
    # TAB 3: POSIÇÕES ABERTAS
    # ========================================================================
    
    with tab3:
        st.subheader("🔓 Posições Abertas")
        
        if not st.session_state.initialized:
            st.warning("⚠️ Componentes não inicializados")
        else:
            positions = st.session_state.order_manager.get_active_positions()
            
            if positions:
                for pos in positions:
                    with st.expander(f"{pos['symbol']} - {pos['side']} {pos['quantity']:.6f}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Entry Price:** ${pos['entry_price']:.2f}")
                            st.write(f"**Stop Loss:** ${pos['stop_loss']:.2f}")
                            st.write(f"**Quantidade Restante:** {pos['quantity_remaining']:.6f}")
                        
                        with col2:
                            st.write(f"**TP1:** {'✅ Atingido' if pos['tp1_hit'] else f'${pos[\"take_profit_1\"]:.2f}'}")
                            st.write(f"**TP2:** ${pos['take_profit_2']:.2f}")
                            st.write(f"**SL Breakeven:** {'✅ Sim' if pos['sl_moved_to_breakeven'] else '❌ Não'}")
                        
                        # Botão para fechar manualmente
                        if st.button(f"Fechar {pos['symbol']}", key=f"close_{pos['symbol']}"):
                            st.session_state.order_manager.close_position_manual(pos['symbol'])
                            st.success(f"Posição {pos['symbol']} fechada!")
                            st.rerun()
            else:
                st.info("📭 Nenhuma posição aberta no momento")
    
    # ========================================================================
    # TAB 4: HISTÓRICO DE TRADES
    # ========================================================================
    
    with tab4:
        st.subheader("📜 Histórico de Trades")
        
        if not st.session_state.initialized:
            st.warning("⚠️ Componentes não inicializados")
        else:
            trade_history = st.session_state.risk_manager.trade_history
            
            if trade_history:
                # Converte para DataFrame
                trades_data = [t.to_dict() for t in trade_history]
                df_trades = pd.DataFrame(trades_data)
                
                # Formata timestamps
                df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'], unit='ms')
                df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'], unit='ms')
                
                # Exibe tabela
                st.dataframe(
                    df_trades[[
                        'trade_id', 'symbol', 'side', 'entry_price', 'exit_price',
                        'pnl', 'pnl_pct', 'reason', 'entry_time', 'exit_time'
                    ]],
                    use_container_width=True
                )
                
                # Botão de exportar
                if st.button("📥 Exportar para CSV"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"trade_history_{timestamp}.csv"
                    df_trades.to_csv(filename, index=False)
                    st.success(f"✅ Exportado: {filename}")
            else:
                st.info("📭 Nenhum trade no histórico")
    
    # ========================================================================
    # TAB 5: BACKTEST
    # ========================================================================
    
    with tab5:
        st.subheader("⚙️ Backtesting")
        
        st.info("🚧 Módulo de backtest será implementado em `run_backtest.py`")
        st.write("Use o comando: `python run_backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31`")
    
    # ========================================================================
    # AUTO REFRESH
    # ========================================================================
    
    if auto_refresh and st.session_state.bot_running:
        time.sleep(refresh_interval)
        st.rerun()


# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    main()