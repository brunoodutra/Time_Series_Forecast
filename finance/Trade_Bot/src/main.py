import time
import sys
import os

from .config import Config
from .logger import setup_logger
from .exchange_client import ExchangeClient
from .state_manager import StateManager
from .signal_reader import SignalReader
from .trade_manager import TradeManager

def main():
    logger = setup_logger()
    logger.info("------------------------------------------------")
    logger.info("   Starting Trade Bot - " + Config.TRADING_MODE)
    logger.info("------------------------------------------------")
    
    try:
        Config.validate()
    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        return

    try:
        exchange = ExchangeClient()
        state = StateManager()
        signals = SignalReader()
        manager = TradeManager(exchange, state)
        
        logger.info(f"Bot initialized. Monitoring: {Config.SYMBOLS}")
        
        while True:
            # logger.debug("Heartbeat check...")
            for symbol in Config.SYMBOLS:
                try:
                    # 1. Sync active trades
                    manager.sync_state(symbol)
                    
                    # 2. Check for new signals
                    latest_signal = signals.get_latest_signal(symbol)
                    
                    if latest_signal:
                        manager.process_signal(symbol, latest_signal)
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            
            # Wait for next cycle
            time.sleep(30)
            
    except KeyboardInterrupt:
        logger.info("Bot stopping...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
