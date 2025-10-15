from optimization.dynamic_params import DynamicParameterManager
from exchange.bybit_ws import BybitWebSocket
import os
from dotenv import load_dotenv

load_dotenv()

# Inicializa componentes
ws = BybitWebSocket(os.getenv("API_KEY"), os.getenv("API_SECRET"), paper_mode=True)
ws.connect()
ws.subscribe_candles("BTCUSDT", ["15", "60"])

# Aguarda dados
import time
time.sleep(10)

# Pega dados
df = ws.get_candles_df("BTCUSDT", "15", limit=200)

# Testa dynamic params
manager = DynamicParameterManager(
    base_atr_multiplier_sl=1.0,
    base_atr_multiplier_tp1=1.0,
    base_atr_multiplier_tp2=2.0,
    base_risk_per_trade=0.01
)

# Analisa e ajusta
conditions = manager.analyze_market_conditions(df)
params = manager.adjust_parameters(conditions)

print("\n📊 Condições:")
print(f"   Volatilidade: {conditions.volatility_regime.value}")
print(f"   Mercado: {conditions.market_regime.value}")

print("\n🔧 Parâmetros Ajustados:")
print(f"   SL: {params.atr_multiplier_sl:.2f}x")
print(f"   TP1: {params.atr_multiplier_tp1:.2f}x")
print(f"   TP2: {params.atr_multiplier_tp2:.2f}x")
print(f"   Risk: {params.risk_per_trade*100:.2f}%")

ws.disconnect()