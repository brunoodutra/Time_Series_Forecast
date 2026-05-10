import json
import os
from datetime import datetime

from .config import Config

class StateManager:
    def __init__(self, filename="active_trades.json", history_filename="order_history.json"):
        self.filepath = os.path.join(Config.DATA_DIR, filename)
        self.history_filepath = os.path.join(Config.DATA_DIR, history_filename)
        self._ensure_files()
        
    def _ensure_files(self):
        if not os.path.exists(self.filepath):
            self.save_state([])
        if not os.path.exists(self.history_filepath):
            self.save_history([])
            
    def load_state(self):
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
            
    def save_state(self, state):
        with open(self.filepath, 'w') as f:
            json.dump(state, f, indent=4)

    def load_history(self):
        try:
            with open(self.history_filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def save_history(self, history):
        with open(self.history_filepath, 'w') as f:
            json.dump(history, f, indent=4)

    def record_event(self, event_type, symbol=None, details=None, trade=None, status=None):
        history = self.load_history()
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "symbol": symbol,
            "status": status,
            "details": details or {},
        }
        if trade:
            event["trade"] = {
                "symbol": trade.get("symbol"),
                "entry_order_id": trade.get("entry_order_id"),
                "side": trade.get("side"),
                "quantity": trade.get("quantity"),
                "entry_price": trade.get("entry_price"),
                "status": trade.get("status"),
                "stop_loss_order_id": trade.get("stop_loss_order_id"),
                "take_profit_order_id": trade.get("take_profit_order_id"),
                "oco_order_id": trade.get("oco_order_id"),
            }
        history.append(event)
        self.save_history(history)
            
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
