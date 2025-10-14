from datetime import datetime
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger('position_tracker')

@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    entry_time: datetime
    tp1_hit: bool = False
    breakeven_moved: bool = False
    unrealized_pnl: float = 0
    current_price: float = 0
    
    def update_pnl(self, current_price):
        self.current_price = current_price
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
    
    def should_move_to_breakeven(self):
        if self.breakeven_moved:
            return False
        if self.side == 'long':
            progress = (self.current_price - self.entry_price) / (self.take_profit_1 - self.entry_price)
        else:
            progress = (self.entry_price - self.current_price) / (self.entry_price - self.take_profit_1)
        return progress >= 0.5

class PositionTracker:
    def __init__(self, client):
        self.client = client
        self.positions = {}
    
    def add_position(self, signal):
        position = Position(
            symbol=signal.symbol, side=signal.side,
            entry_price=signal.entry_price, size=signal.position_size,
            stop_loss=signal.stop_loss, take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2, entry_time=datetime.now()
        )
        self.positions[signal.symbol] = position
        logger.info(f"Position tracked: {signal.side} {signal.symbol}")
    
    def update_positions(self):
        for symbol in list(self.positions.keys()):
            ticker = self.client.get_ticker(symbol)
            if ticker:
                self.positions[symbol].update_pnl(ticker['last'])
    
    def get_position(self, symbol):
        return self.positions.get(symbol)
    
    def get_all_positions(self):
        return list(self.positions.values())