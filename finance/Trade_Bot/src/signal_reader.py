import logging
from urllib.parse import urlencode

import requests

from .config import Config

class SignalReader:
    def __init__(self):
        self.logger = logging.getLogger("TradeBot.SignalReader")

    def _normalize_crypto(self, symbol):
        if symbol.endswith("USDT"):
            return symbol[:-4]
        return symbol

    def _build_url(self, endpoint, params):
        query = urlencode(params)
        return f"{Config.API_BASE_URL.rstrip('/')}/{endpoint}?{query}"

    def _request_json(self, endpoint, params, symbol=None):
        url = self._build_url(endpoint, params)
        try:
            response = requests.get(url, timeout=Config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                if symbol:
                    self.logger.warning("API retornou erro para %s (%s): %s", endpoint, symbol, payload["error"])
                else:
                    self.logger.warning("API retornou erro para %s: %s", endpoint, payload["error"])
                return None
            return payload
        except Exception as e:
            if symbol:
                self.logger.error("Falha ao consultar %s (%s): %s", url, symbol, e)
            else:
                self.logger.error("Falha ao consultar %s: %s", url, e)
            return None

    def get_latest_signal(self, symbol):
        crypto = self._normalize_crypto(symbol)
        recommendation = self._request_json(
            "last_recommendation",
            {"model_name": Config.MODEL_NAME, "crypto": crypto},
            symbol=symbol,
        )
        if not recommendation:
            return None

        recommendation_value = recommendation.get("recommendation")
        target_stop = None
        if recommendation_value and recommendation_value != "Hold":
            target_stop = self._request_json(
                "last_target_stop",
                {"model_name": Config.MODEL_NAME, "crypto": crypto, "profile": "conservative"},
                symbol=symbol,
            )

        date_value = recommendation.get("Date")
        time_value = recommendation.get("Time")
        timestamp = f"{date_value} {time_value}".strip() if date_value or time_value else None

        signal_gains = None
        if target_stop:
            signal_gains = {
                "Target": target_stop.get("target"),
                "StopLoss": target_stop.get("stop_loss"),
            }

        try:
            price = float(recommendation.get("Price", 0.0))
        except (TypeError, ValueError):
            price = 0.0

        try:
            percentage = float(recommendation.get("percentage", 0.0))
        except (TypeError, ValueError):
            percentage = 0.0

        return {
            "timestamp": timestamp,
            "recommendation": recommendation.get("recommendation"),
            "percentage": percentage,
            "price": price,
            "signal_gains": signal_gains,
        }
