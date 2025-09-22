"""
Shared Logging Utilities

Centralized logging configuration and utilities to reduce redundancy across modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if logs directory exists)
        logs_dir = Path("logs")
        if logs_dir.exists():
            file_handler = logging.FileHandler(logs_dir / "system.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup global logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file or logs_dir / "system.log")
        ]
    )


# Convenience function for quick logger access
def get_module_logger(module_name: str) -> logging.Logger:
    """Get logger for a specific module"""
    return get_logger(module_name)
