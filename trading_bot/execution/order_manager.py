from config.settings import Settings
from utils.logger import setup_logger
from utils.helpers import calculate_position_size

logger = setup_logger('order_manager')

class OrderManager:
    def __init__(self, client):
        self.client = client
    
    def execute_signal(self, signal, capital):
        symbol = signal.symbol
        position_size = calculate_position_size(
            capital, Settings.RISK_PER_TRADE,
            signal.entry_price, signal.stop_loss, Settings.LEVERAGE
        )
        if position_size == 0:
            return False
        
        signal.position_size = position_size
        self.client.set_leverage(symbol, Settings.LEVERAGE)
        
        entry_side = 'buy' if signal.side == 'long' else 'sell'
        order = self.client.create_market_order(symbol, entry_side, position_size)
        
        if order:
            logger.info(f"✓ Order executed: {entry_side} {position_size} {symbol}")
            return True
        return False