def format_number(value, decimals=2):
    return f"{value:,.{decimals}f}"

def calculate_pnl(entry_price, exit_price, quantity, side, fees=0.0):
    if side.lower() == 'long':
        pnl = (exit_price - entry_price) * quantity
    else:
        pnl = (entry_price - exit_price) * quantity
    return pnl - fees

def calculate_position_size(capital, risk_pct, entry_price, stop_loss_price, leverage=1):
    risk_amount = capital * (risk_pct / 100)
    price_diff = abs(entry_price - stop_loss_price)
    if price_diff == 0:
        return 0
    position_size = (risk_amount / price_diff) * leverage
    return position_size