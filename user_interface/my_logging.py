import atexit
import logging
import logging.config
import os
import queue
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler
from pathlib import Path
from functools import partial

DEFAULT_LOGGER_NAME = "plastinka_photoassistant"
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

# Singleton objects to ensure idempotency
_log_queue: queue.Queue | None = None
_queue_listener: QueueListener | None = None

# Default error path - will be updated when setup_logging is called
ERROR_PATH = Path("logs/errors")

# Get configuration from environment variables or use defaults
log_level = os.getenv('PLASTINKA_LOG_LEVEL', 'INFO')
log_dir = os.getenv('PLASTINKA_LOG_DIR', 'logs')


def _setup_logging(
    log_level: str = 'INFO',
    log_dir: str | Path | None = None,
    logger_name: str = DEFAULT_LOGGER_NAME,
) -> None:
    """
    Configure root logger with async queue-based logging handlers.

    Args:
        log_level: The logging level (e.g., 'INFO', 'DEBUG', 'ERROR').
        log_dir: Directory to save log files. If None, only console logging is used.
        logger_name: The root logger name for the application.
    """
    # Update ERROR_PATH first
    _update_error_path(log_dir)
    
    root_logger = logging.getLogger(logger_name)
    root_logger.setLevel(log_level.upper())

    # Remove all existing handlers to prevent duplicates
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    
    # Also clear handlers for the root of all loggers if we are configuring our main logger
    if logger_name == DEFAULT_LOGGER_NAME:
        for h in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(h)

    global _log_queue, _queue_listener
    if _log_queue is None and _queue_listener is None:
        handlers = _build_async_handlers(log_level, log_dir)
        _log_queue = queue.Queue(-1)
        queue_handler = QueueHandler(_log_queue)
        root_logger.addHandler(queue_handler)

        _queue_listener = QueueListener(
            _log_queue, *handlers, respect_handler_level=True
        )
        _queue_listener.start()

        def _shutdown_logging():
            if _queue_listener:
                _queue_listener.stop()
        
        atexit.register(_shutdown_logging)

def _build_async_handlers(
    log_level: str, log_dir: str | Path | None
) -> list[logging.Handler]:
    """Build handlers for async, queue-based logging."""
    formatter = logging.Formatter(_LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level.upper())
    
    handlers = [console_handler]
    
    # File handler (if log_dir is provided)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main application log
        file_path = log_dir / "photoassistant.log"
        file_handler = TimedRotatingFileHandler(
            filename=file_path,
            when="D",
            interval=1,
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level.upper())
        handlers.append(file_handler)
        
        # Error log (separate file for errors)
        error_path = log_dir / "errors"
        error_path.mkdir(exist_ok=True)
        error_file_path = error_path / "error_log"
        error_handler = TimedRotatingFileHandler(
            filename=error_file_path,
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        handlers.append(error_handler)
        
    return handlers

def get_logger(module_name: str = None) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    Args:
        module_name: Name of the module requesting the logger. 
                    If None, returns the root logger.
    
    Returns:
        Logger instance configured for the module.
    """
    if module_name:
        return logging.getLogger(DEFAULT_LOGGER_NAME).getChild(module_name)
    else:
        return logging.getLogger(DEFAULT_LOGGER_NAME)

def _update_error_path(log_dir: str | Path | None):
    """Update the global ERROR_PATH when setup_logging is called."""
    global ERROR_PATH
    if log_dir:
        ERROR_PATH = Path(log_dir) / "errors"
    else:
        ERROR_PATH = Path("logs/errors")


setup_logging = partial(_setup_logging,
    log_level=log_level,
    log_dir=log_dir,
    logger_name=DEFAULT_LOGGER_NAME
)

