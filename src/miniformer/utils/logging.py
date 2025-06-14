import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: Optional[int] = None,
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Get a logger with specified name and configuration
    
    Args:
        name: Logger name
        level: Logging level (defaults to INFO)
        format: Log format string
        
    Returns:
        logger: Configured logger
    """
    if level is None:
        level = logging.INFO
        
    logger = logging.getLogger(name)
    
    # Only configure the logger if it hasn't been configured before
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create handlers
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(format)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
    return logger


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> None:
    """
    Setup global logging configuration
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format: Log format string
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
