import logging
import os
from datetime import datetime
from .config import Config

def setup_logger(name="TradeBot"):
    log_dir = Config.LOG_DIR
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"bot_{timestamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
