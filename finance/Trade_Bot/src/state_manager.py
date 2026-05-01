import json
import os
from .config import Config

class StateManager:
    def __init__(self, filename="active_trades.json"):
        self.filepath = os.path.join(Config.DATA_DIR, filename)
        self._ensure_file()
        
    def _ensure_file(self):
        if not os.path.exists(self.filepath):
            self.save_state([])
            
    def load_state(self):
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
            
    def save_state(self, state):
        with open(self.filepath, 'w') as f:
            json.dump(state, f, indent=4)
            
    def get_active_trade(self, symbol):
        """Returns the active trade for a symbol if exists."""
        trades = self.load_state()
        for trade in trades:
            if trade.get('symbol') == symbol and trade.get('status') in ['OPEN', 'PENDING', 'PARTIALLY_FILLED']:
                return trade
        return None
        
    def update_trade(self, trade_data):
        """Updates an existing trade or adds a new one."""
        trades = self.load_state()
        updated = False
        
        # If trade has an ID, use it for matching
        trade_id = trade_data.get('entry_order_id')
        symbol = trade_data.get('symbol')
        
        for i, trade in enumerate(trades):
            # Match by order ID if available, otherwise by symbol for active trades
            if (trade_id and trade.get('entry_order_id') == trade_id) or \
               (not trade_id and trade.get('symbol') == symbol and trade.get('status') in ['OPEN', 'PENDING']):
                trades[i] = trade_data
                updated = True
                break
        
        if not updated:
            trades.append(trade_data)
            
        self.save_state(trades)
        
    def close_trade(self, symbol, exit_price=None, exit_time=None):
        """Marks a trade as CLOSED."""
        trades = self.load_state()
        for trade in trades:
            if trade.get('symbol') == symbol and trade.get('status') in ['OPEN', 'PENDING', 'PARTIALLY_FILLED']:
                trade['status'] = 'CLOSED'
                if exit_price:
                    trade['exit_price'] = exit_price
                if exit_time:
                    trade['exit_time'] = exit_time
        self.save_state(trades)
