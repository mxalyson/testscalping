class PerformanceMetrics:
    @staticmethod
    def calculate(trades, equity_curve):
        return {'total_trades': len(trades), 'win_rate': 0}
    
    @staticmethod
    def print_summary(metrics):
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total trades: {metrics.get('total_trades', 0)}")
        print("="*50 + "\n")