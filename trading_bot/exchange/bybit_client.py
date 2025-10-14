import ccxt
import time
from datetime import datetime
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger('bybit_client')

class BybitClient:
    def __init__(self, paper_mode=True):
        self.paper_mode = paper_mode
        config = Settings.get_exchange_config()
        self.exchange = ccxt.bybit(config)
        self.paper_balance = Settings.INITIAL_CAPITAL
        self.paper_positions = {}
        self.paper_trades = []
        logger.info(f"Bybit client initialized - Paper: {paper_mode}")
    
    def get_balance(self):
        if self.paper_mode:
            return {'total': self.paper_balance, 'free': self.paper_balance, 'used': 0}
        try:
            balance = self.exchange.fetch_balance()
            return {'total': balance['total'].get('USDT', 0), 
                    'free': balance['free'].get('USDT', 0), 
                    'used': balance['used'].get('USDT', 0)}
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {'total': 0, 'free': 0, 'used': 0}
    
    def set_leverage(self, symbol, leverage):
        if self.paper_mode:
            logger.info(f"Paper: Set leverage {leverage}x for {symbol}")
            return True
        try:
            self.exchange.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return False
    
    def create_market_order(self, symbol, side, amount, reduce_only=False):
        if self.paper_mode:
            ticker = self.get_ticker(symbol)
            price = ticker['last'] if ticker else 0
            order = {
                'id': f"paper_{int(time.time()*1000)}",
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'status': 'filled'
            }
            logger.info(f"Paper order: {side} {amount} {symbol} @ {price}")
            return order
        try:
            order = self.exchange.create_order(symbol, 'market', side, amount)
            return order
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None
    
    def get_ticker(self, symbol):
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return None
    
    def get_position(self, symbol):
        return self.paper_positions.get(symbol) if self.paper_mode else None
    
    def update_paper_positions(self):
        for symbol in list(self.paper_positions.keys()):
            ticker = self.get_ticker(symbol)
            if ticker:
                pos = self.paper_positions[symbol]
                current_price = ticker['last']
                if pos['side'] == 'long':
                    pos['unrealized_pnl'] = (current_price - pos['entry_price']) * pos['amount']
                else:
                    pos['unrealized_pnl'] = (pos['entry_price'] - current_price) * pos['amount']