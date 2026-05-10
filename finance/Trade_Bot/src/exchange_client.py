import ccxt
import logging
import os
from datetime import datetime

from .config import Config

class ExchangeClient:
    def __init__(self):
        self.logger = logging.getLogger("TradeBot.Exchange")
        self.is_paper = Config.TRADING_MODE == "PAPER"
        self.paper_balance = float(os.getenv("PAPER_BALANCE", "1000"))
        self.market_mode = (getattr(Config, "MARKET_MODE", "FUTURES") or "FUTURES").upper()

        exchange_config = {
            'enableRateLimit': True,
        }
        if not self.is_paper and Config.API_KEY and Config.SECRET_KEY:
            exchange_config['apiKey'] = Config.API_KEY
            exchange_config['secret'] = Config.SECRET_KEY

        if self.market_mode == "SPOT":
            self.exchange = ccxt.binance(exchange_config)
            self.market_type = "spot"
        else:
            self.exchange = ccxt.binanceusdm(exchange_config)
            self.market_type = "futures-usdm"

        try:
            self.exchange.load_markets()
            self.logger.info("Exchange configurado para: %s", self.market_type)
        except Exception as e:
            self.logger.warning("Falha ao carregar mercados da Binance: %s", e)

        if self.is_paper:
            self.logger.info("Running in PAPER mode (sem envio de ordens reais)")

    def _normalize_symbol(self, symbol):
        if not isinstance(symbol, str):
            return symbol
        if "/" in symbol:
            return symbol
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            if self.market_type == "spot":
                return f"{base}/USDT"
            return f"{base}/USDT:USDT"
        return symbol

    def validate_connection(self):
        if self.is_paper:
            return True

        try:
            self.exchange.load_markets()
            balance = self.exchange.fetch_balance()
            usdt_info = balance.get('USDT', {})
            self.logger.info(
                "Conexao Binance %s validada. USDT free=%s total=%s",
                self.market_type,
                usdt_info.get('free'),
                usdt_info.get('total'),
            )
            return True
        except ccxt.AuthenticationError as e:
            self.logger.error(
                "Falha de autenticacao na Binance %s. Verifique API key, secret, permissoes e whitelist de IP: %s",
                self.market_type,
                e,
            )
            return False
        except Exception as e:
            self.logger.error("Falha ao validar conexao com Binance %s: %s", self.market_type, e)
            return False
            
    def get_balance(self):
        """Returns USDT free balance."""
        if self.is_paper:
            return float(self.paper_balance)
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return 0.0

    def get_ticker(self, symbol):
        """Returns current ticker data."""
        try:
            return self.exchange.fetch_ticker(self._normalize_symbol(symbol))
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

    def create_market_order(self, symbol, side, amount):
        """Creates a MARKET order."""
        if self.is_paper:
            ticker = self.get_ticker(symbol) or {}
            price = float(ticker.get("last") or ticker.get("close") or 0.0)
            try:
                amount = float(self.exchange.amount_to_precision(self._normalize_symbol(symbol), amount))
            except Exception:
                amount = float(amount)

            order_id = f"paper_{symbol}_{side}_{datetime.utcnow().timestamp()}"
            self.logger.info("PAPER order: market %s %s amount=%s price=%s id=%s", side, symbol, amount, price, order_id)
            return {
                "id": order_id,
                "symbol": symbol,
                "type": "market",
                "side": side,
                "amount": amount,
                "average": price,
                "price": price,
                "status": "FILLED",
            }
        try:
            # amount must be precision adjusted
            normalized_symbol = self._normalize_symbol(symbol)
            amount = self.exchange.amount_to_precision(normalized_symbol, amount)
            order = self.exchange.create_order(normalized_symbol, 'market', side, amount)
            self.logger.info(f"Market {side} order created for {symbol}: {order['id']}")
            return order
        except Exception as e:
            self.logger.error(f"Error creating market order: {e}")
            raise e

    def create_stop_loss_order(self, symbol, side, amount, stop_price):
        """Creates a STOP_MARKET order."""
        if self.market_type == "spot":
            raise NotImplementedError("Use create_oco_order for spot protection orders.")
        if self.is_paper:
            try:
                amount = float(self.exchange.amount_to_precision(self._normalize_symbol(symbol), amount))
            except Exception:
                amount = float(amount)
            try:
                stop_price = float(self.exchange.price_to_precision(self._normalize_symbol(symbol), stop_price))
            except Exception:
                stop_price = float(stop_price)

            order_id = f"paper_sl_{symbol}_{side}_{datetime.utcnow().timestamp()}"
            self.logger.info("PAPER order: stop_loss %s %s amount=%s stop=%s id=%s", side, symbol, amount, stop_price, order_id)
            return {"id": order_id, "symbol": symbol, "type": "STOP_MARKET", "side": side, "amount": amount, "stopPrice": stop_price, "status": "OPEN"}
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            amount = self.exchange.amount_to_precision(normalized_symbol, amount)
            price = self.exchange.price_to_precision(normalized_symbol, stop_price)
            
            params = {'stopPrice': price}
            order = self.exchange.create_order(normalized_symbol, 'STOP_MARKET', side, amount, params=params)
            self.logger.info(f"Stop loss order created for {symbol} at {price}: {order['id']}")
            return order
        except Exception as e:
            self.logger.error(f"Error creating stop loss order: {e}")
            return None

    def create_take_profit_order(self, symbol, side, amount, tp_price):
        """Creates a TAKE_PROFIT_MARKET order."""
        if self.market_type == "spot":
            raise NotImplementedError("Use create_oco_order for spot protection orders.")
        if self.is_paper:
            try:
                amount = float(self.exchange.amount_to_precision(self._normalize_symbol(symbol), amount))
            except Exception:
                amount = float(amount)
            try:
                tp_price = float(self.exchange.price_to_precision(self._normalize_symbol(symbol), tp_price))
            except Exception:
                tp_price = float(tp_price)

            order_id = f"paper_tp_{symbol}_{side}_{datetime.utcnow().timestamp()}"
            self.logger.info("PAPER order: take_profit %s %s amount=%s tp=%s id=%s", side, symbol, amount, tp_price, order_id)
            return {"id": order_id, "symbol": symbol, "type": "TAKE_PROFIT_MARKET", "side": side, "amount": amount, "stopPrice": tp_price, "status": "OPEN"}
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            amount = self.exchange.amount_to_precision(normalized_symbol, amount)
            price = self.exchange.price_to_precision(normalized_symbol, tp_price)
            
            params = {'stopPrice': price}
            order = self.exchange.create_order(normalized_symbol, 'TAKE_PROFIT_MARKET', side, amount, params=params)
            self.logger.info(f"Take profit order created for {symbol} at {price}: {order['id']}")
            return order
        except Exception as e:
            self.logger.error(f"Error creating take profit order: {e}")
            return None
            
    def get_order(self, symbol, order_id):
        """Fetches order status."""
        if self.is_paper:
            if isinstance(order_id, str) and order_id.startswith("paper_"):
                if "_sl_" in order_id or order_id.startswith("paper_sl_"):
                    return {"id": order_id, "symbol": symbol, "status": "OPEN"}
                if "_tp_" in order_id or order_id.startswith("paper_tp_"):
                    return {"id": order_id, "symbol": symbol, "status": "OPEN"}
                return {"id": order_id, "symbol": symbol, "status": "FILLED"}
        try:
            return self.exchange.fetch_order(order_id, self._normalize_symbol(symbol))
        except Exception as e:
            self.logger.error(f"Error fetching order {order_id}: {e}")
            return None
            
    def cancel_order(self, symbol, order_id):
        """Cancels an order."""
        if self.is_paper:
            self.logger.info("PAPER cancel order %s (%s)", order_id, symbol)
            return True
        try:
            self.exchange.cancel_order(order_id, self._normalize_symbol(symbol))
            self.logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def create_oco_order(self, symbol, side, amount, take_profit_price, stop_loss_price):
        if self.market_type != "spot":
            raise ValueError("OCO é suportado apenas em SPOT nesta implementacao.")
        if self.is_paper:
            order_id = f"paper_oco_{symbol}_{side}_{datetime.utcnow().timestamp()}"
            self.logger.info(
                "PAPER order: oco %s %s amount=%s tp=%s sl=%s id=%s",
                side,
                symbol,
                amount,
                take_profit_price,
                stop_loss_price,
                order_id,
            )
            return {"id": order_id, "symbol": symbol, "type": "OCO", "side": side, "amount": amount, "status": "OPEN"}

        normalized_symbol = self._normalize_symbol(symbol)
        amount = self.exchange.amount_to_precision(normalized_symbol, amount)
        tp_price = self.exchange.price_to_precision(normalized_symbol, take_profit_price)
        sl_stop = self.exchange.price_to_precision(normalized_symbol, stop_loss_price)
        sl_limit = self.exchange.price_to_precision(normalized_symbol, stop_loss_price)

        params = {
            "stopPrice": sl_stop,
            "stopLimitPrice": sl_limit,
            "stopLimitTimeInForce": "GTC",
        }
        order = self.exchange.create_order(normalized_symbol, "OCO", side, amount, tp_price, params=params)
        self.logger.info("OCO order created for %s: %s", symbol, order.get("id"))
        return order
            
    def get_position(self, symbol):
        """Fetches current open position for symbol."""
        if self.market_type == "spot":
            try:
                normalized_symbol = self._normalize_symbol(symbol)
                base_asset = normalized_symbol.split("/")[0]
                balance = self.exchange.fetch_balance()
                asset_info = balance.get(base_asset, {})
                return {"symbol": normalized_symbol, "asset": base_asset, "free": asset_info.get("free"), "total": asset_info.get("total")}
            except Exception as e:
                self.logger.error(f"Error fetching spot balance for {symbol}: {e}")
                return None
        try:
            positions = self.exchange.fetch_positions([self._normalize_symbol(symbol)])
            for pos in positions:
                if pos['symbol'] == self._normalize_symbol(symbol):
                    return pos
            return None
        except Exception as e:
            self.logger.error(f"Error fetching position for {symbol}: {e}")
            return None
