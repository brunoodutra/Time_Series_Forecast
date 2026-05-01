import logging
import time
from datetime import datetime
from .config import Config

class TradeManager:
    def __init__(self, exchange, state_manager):
        self.exchange = exchange
        self.state = state_manager
        self.logger = logging.getLogger("TradeBot.TradeManager")

    def _is_paper_order_id(self, order_id):
        return isinstance(order_id, str) and order_id.startswith("paper_")

    def _has_stale_paper_state(self, trade):
        return any(
            self._is_paper_order_id(trade.get(key))
            for key in ("entry_order_id", "stop_loss_order_id", "take_profit_order_id")
        )

    def _resolve_order_fill_price(self, order, trade):
        candidates = [
            order.get('average'),
            order.get('price'),
            order.get('last'),
            order.get('stopPrice'),
            trade.get('entry_price'),
            trade.get('signal_data', {}).get('price'),
        ]
        for value in candidates:
            try:
                if value is not None and value != "":
                    return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0
        
    def sync_state(self, symbol):
        """Syncs local state with exchange state."""
        trade = self.state.get_active_trade(symbol)
        if not trade:
            return

        if not getattr(self.exchange, "is_paper", False) and self._has_stale_paper_state(trade):
            self.logger.warning(
                "Estado PAPER detectado em modo REAL para %s. Encerrando trade salvo localmente antes de sincronizar com a Binance.",
                symbol,
            )
            self.state.close_trade(symbol, exit_time=datetime.now().isoformat())
            return
            
        # 1. Check Entry Order
        if trade['status'] == 'PENDING':
            order = self.exchange.get_order(symbol, trade['entry_order_id'])
            if not order:
                return

            if order['status'] == 'FILLED':
                self.logger.info(f"Entry order filled for {symbol}")
                trade['status'] = 'OPEN'
                trade['entry_price'] = self._resolve_order_fill_price(order, trade)
                self.state.update_trade(trade)
                
                # Place Protection Orders immediately
                self.place_protection_orders(trade)
                
            elif order['status'] in ['CANCELED', 'REJECTED', 'EXPIRED']:
                self.logger.warning(f"Entry order {order['status']} for {symbol}")
                trade['status'] = 'CLOSED'
                self.state.update_trade(trade)
                
        # 2. Monitor Open Trade (SL/TP)
        elif trade['status'] == 'OPEN':
            # Check Stop Loss
            if trade.get('stop_loss_order_id'):
                sl_order = self.exchange.get_order(symbol, trade['stop_loss_order_id'])
                if sl_order and sl_order['status'] == 'FILLED':
                    self.logger.info(f"STOP LOSS triggered for {symbol}")
                    self.close_trade_cleanup(trade, reason="STOP_LOSS")
                    return

            # Check Take Profit
            if trade.get('take_profit_order_id'):
                tp_order = self.exchange.get_order(symbol, trade['take_profit_order_id'])
                if tp_order and tp_order['status'] == 'FILLED':
                    self.logger.info(f"TAKE PROFIT triggered for {symbol}")
                    self.close_trade_cleanup(trade, reason="TAKE_PROFIT")
                    return

    def place_protection_orders(self, trade):
        """Places Stop Loss and Take Profit orders."""
        symbol = trade['symbol']
        side = 'sell' if trade['side'] == 'buy' else 'buy'
        quantity = trade['quantity']
        entry_price = trade['entry_price']
        
        # Determine levels from Signal Data or Defaults
        sl_price = 0.0
        tp_price = 0.0
        
        signal_gains = trade.get('signal_data', {}).get('signal_gains')
        
        if signal_gains:
            # Assuming structure from bot_advisor_trade.ipynb logic
            # It might be a dict with 'StopLoss' and 'Target'
            # Note: The notebook logic for SignalGains wasn't fully visible, 
            # but usually it's calculated. I'll add safe fallbacks.
            try:
                if trade['side'] == 'buy':
                    sl_price = float(signal_gains.get('StopLoss', entry_price * 0.98))
                    tp_price = float(signal_gains.get('Target', entry_price * 1.04)) # First target
                else:
                    sl_price = float(signal_gains.get('StopLoss', entry_price * 1.02))
                    tp_price = float(signal_gains.get('Target', entry_price * 0.96))
            except:
                self.logger.warning("Failed to parse SignalGains, using defaults")
                sl_price = entry_price * 0.98 if trade['side'] == 'buy' else entry_price * 1.02
                tp_price = entry_price * 1.04 if trade['side'] == 'buy' else entry_price * 0.96
        else:
             # Default 2% SL, 4% TP
            sl_price = entry_price * 0.98 if trade['side'] == 'buy' else entry_price * 1.02
            tp_price = entry_price * 1.04 if trade['side'] == 'buy' else entry_price * 0.96

        # Create Orders
        sl_order = self.exchange.create_stop_loss_order(symbol, side, quantity, sl_price)
        if sl_order:
            trade['stop_loss_order_id'] = sl_order['id']
            
        tp_order = self.exchange.create_take_profit_order(symbol, side, quantity, tp_price)
        if tp_order:
            trade['take_profit_order_id'] = tp_order['id']
            
        self.state.update_trade(trade)

    def close_trade_cleanup(self, trade, reason="MANUAL"):
        """Cancels remaining orders and marks trade as closed."""
        symbol = trade['symbol']
        
        # Cancel SL if TP hit, or vice versa
        if trade.get('stop_loss_order_id'):
            self.exchange.cancel_order(symbol, trade['stop_loss_order_id'])
            
        if trade.get('take_profit_order_id'):
            self.exchange.cancel_order(symbol, trade['take_profit_order_id'])
            
        self.state.close_trade(symbol, exit_time=datetime.now().isoformat())
        self.logger.info(f"Trade for {symbol} closed. Reason: {reason}")

    def close_position_market(self, trade):
        """Manually closes a position at market price."""
        symbol = trade['symbol']
        side = 'sell' if trade['side'] == 'buy' else 'buy'
        quantity = trade['quantity']
        
        self.exchange.create_market_order(symbol, side, quantity)
        self.close_trade_cleanup(trade, reason="REVERSAL")

    def process_signal(self, symbol, signal_data):
        """Main decision logic."""
        if not signal_data:
            return

        active_trade = self.state.get_active_trade(symbol)
        
        recommendation = signal_data['recommendation']
        confidence = signal_data['percentage']
        timestamp = signal_data['timestamp']
        
        # Ignore old signals (e.g., > 2 hours old) - configurable
        # TODO: Implement time check
        
        if confidence < Config.CONFIDENCE_THRESHOLD:
            return
            
        if active_trade:
            # Check for reversal
            if (active_trade['side'] == 'buy' and recommendation == 'Sell') or \
               (active_trade['side'] == 'sell' and recommendation == 'Buy'):
                self.logger.info(f"Reversal signal for {symbol}. Closing position.")
                self.close_position_market(active_trade)
                # Optionally open new trade here
        else:
            # Only open if signal matches trend
            if recommendation == 'Buy':
                self.open_position(symbol, 'buy', signal_data)
            elif recommendation == 'Sell':
                self.open_position(symbol, 'sell', signal_data)

    def open_position(self, symbol, side, signal_data):
        balance = self.exchange.get_balance()
        price = signal_data['price']
        
        if balance < 10: # Min 10 USDT
            self.logger.warning("Insufficient balance to trade")
            return

        # Calculate Position Size
        # Simple Logic: Use X% of balance
        risk_amount = balance * Config.RISK_PER_TRADE
        quantity = (risk_amount * Config.LEVERAGE) / price
        
        self.logger.info(f"Opening {side} position for {symbol}. Qty: {quantity}")
        
        order = self.exchange.create_market_order(symbol, side, quantity)
        if order:
            trade = {
                'symbol': symbol,
                'entry_order_id': order['id'],
                'side': side,
                'quantity': quantity,
                'entry_price': price, # Approximate, updated when filled
                'status': 'PENDING',
                'timestamp': datetime.now().isoformat(),
                'signal_data': signal_data
            }
            self.state.update_trade(trade)
