import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("BINANCE_API_KEY")
    SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
    TRADING_MODE = os.getenv("TRADING_MODE", "PAPER").upper()
    MARKET_MODE = os.getenv("MARKET_MODE", "FUTURES").upper()  # FUTURES | SPOT
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    API_BASE_URL = os.getenv("RECOMMENDATION_API_URL", "http://127.0.0.1:8000")
    MODEL_NAME = os.getenv("RECOMMENDATION_MODEL_NAME", "CNN")
    API_TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "10"))

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    SYMBOLS = ["BTCUSDT", "ETHUSDT","XRPUSDT", "SOLUSDT", "ADAUSDT"]
    TIMEFRAME = "4h"
    # Confianca minima do modelo para aceitar um sinal e tentar operar.
    CONFIDENCE_THRESHOLD = 0.60
    # Percentual do saldo usado em cada operacao; 0.08 = 8% do capital disponivel.
    RISK_PER_TRADE = 0.25
    # Multiplicador de exposicao no futures; 1 significa sem alavancagem adicional.
    LEVERAGE = 10

    @staticmethod
    def validate():
        if Config.TRADING_MODE != "PAPER" and (not Config.API_KEY or not Config.SECRET_KEY):
            raise ValueError("API_KEY and SECRET_KEY must be set in .env file")
