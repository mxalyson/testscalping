from datetime import datetime
from dataclasses import dataclass
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger('risk_manager')

@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    timestamp: datetime
    
    @property
    def is_winner(self):
        return self.pnl > 0

class RiskManager:
    def __init__(self):
        self.trades_today = []
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        logger.info("Risk manager initialized")
    
    def can_open_position(self, current_positions, capital):
        if self.circuit_breaker_active:
            return False, "Circuit breaker active"
        if current_positions >= Settings.MAX_POSITIONS:
            return False, f"Max positions ({Settings.MAX_POSITIONS})"
        if capital > 0:
            drawdown_pct = ((Settings.INITIAL_CAPITAL - capital) / Settings.INITIAL_CAPITAL) * 100
            if drawdown_pct >= Settings.MAX_DAILY_DRAWDOWN:
                self.circuit_breaker_active = True
                return False, f"Max drawdown {drawdown_pct:.2f}%"
        if self.consecutive_losses >= Settings.MAX_CONSECUTIVE_LOSSES:
            self.circuit_breaker_active = True
            return False, "Max consecutive losses"
        return True, "OK"
    
    def record_trade(self, trade):
        self.trades_today.append(trade)
        self.daily_pnl += trade.pnl
        if trade.is_winner:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
    
    def get_daily_stats(self):
        if not self.trades_today:
            return {
                'total_trades': 0, 'winners': 0, 'losers': 0,
                'win_rate': 0, 'total_pnl': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0
            }
        winners = [t for t in self.trades_today if t.is_winner]
        losers = [t for t in self.trades_today if not t.is_winner]
        total_wins = sum(t.pnl for t in winners)
        total_losses = abs(sum(t.pnl for t in losers))
        return {
            'total_trades': len(self.trades_today),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': (len(winners) / len(self.trades_today)) * 100,
            'total_pnl': self.daily_pnl,
            'avg_win': total_wins / len(winners) if winners else 0,
            'avg_loss': total_losses / len(losers) if losers else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else 0
        }