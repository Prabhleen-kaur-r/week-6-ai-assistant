"""Logging configuration for the application."""

import logging
import sys
from typing import Optional
from datetime import datetime


def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    
    # Set logging levels for third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    
    # Create application logger
    logger = get_logger("app")
    logger.info(f"Logging initialized at level {level}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: The name of the logger (typically __name__)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


class LoggingContext:
    """
    Context manager for adding extra context to logs.
    
    Example:
        with LoggingContext(user_id="123", session="abc"):
            logger.info("Processing request")
    """
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.old_factory = None
        
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.kwargs.items():
                setattr(record, key, value)
            return record
            
        logging.setLogRecordFactory(factory)
        return self
        
    def __exit__(self, *args):
        logging.setLogRecordFactory(self.old_factory)