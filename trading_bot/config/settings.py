import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Settings:
    BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
    BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')
    PAPER_MODE = os.getenv('PAPER_MODE', 'true').lower() == 'true'
    USE_TESTNET = os.getenv('USE_TESTNET', 'true').lower() == 'true'
    LEVERAGE = int(os.getenv('LEVERAGE', '10'))
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '1.0'))
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '3'))
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '10000'))
    SL_ATR_MULTIPLIER = float(os.getenv('SL_ATR_MULTIPLIER', '2.0'))
    TP1_ATR_MULTIPLIER = float(os.getenv('TP1_ATR_MULTIPLIER', '1.0'))
    TP2_ATR_MULTIPLIER = float(os.getenv('TP2_ATR_MULTIPLIER', '2.0'))
    TP1_PERCENTAGE = float(os.getenv('TP1_PERCENTAGE', '50'))
    BREAKEVEN_PERCENTAGE = float(os.getenv('BREAKEVEN_PERCENTAGE', '50'))
    MAX_DAILY_DRAWDOWN = float(os.getenv('MAX_DAILY_DRAWDOWN', '5.0'))
    MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', '5'))
    TRADING_SYMBOLS = os.getenv('TRADING_SYMBOLS', 'BTCUSDT,ETHUSDT,SOLUSDT').split(',')
    PRIMARY_TIMEFRAME = int(os.getenv('PRIMARY_TIMEFRAME', '15'))
    CONFIRMATION_TIMEFRAMES = [int(tf) for tf in os.getenv('CONFIRMATION_TIMEFRAMES', '60,120,240').split(',')]
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'trading_bot.log')
    
    @classmethod
    def validate(cls):
        if not cls.PAPER_MODE and not cls.USE_TESTNET:
            if not cls.BYBIT_API_KEY or not cls.BYBIT_API_SECRET:
                raise ValueError("API credentials required")
        return True
    
    @classmethod
    def get_exchange_config(cls):
        config = {
            'apiKey': cls.BYBIT_API_KEY,
            'secret': cls.BYBIT_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        }
        if cls.USE_TESTNET:
            config['urls'] = {
                'api': {
                    'public': 'https://api-testnet.bybit.com',
                    'private': 'https://api-testnet.bybit.com'
                }
            }
        return config