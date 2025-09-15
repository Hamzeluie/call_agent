import logging
import json
import os
import sys
from datetime import datetime
from logging.handlers import MemoryHandler # Import MemoryHandler
from collections import deque # For a fixed-size buffer for the API
from utils import get_env_variable

# Environment variables for logging `configuration`
LOG_LEVEL_ENV = get_env_variable("LOG_LEVEL").upper()
LOG_FORMAT_ENV = get_env_variable("LOG_FORMAT").lower() # "json" or "text"
APP_LOGGER_NAME = get_env_variable("APP_LOGGER_NAME")
MEMORY_LOG_CAPACITY = get_env_variable("MEMORY_LOG_CAPACITY", var_type=int)

# This deque will store formatted log messages for API retrieval
# It's a simpler approach than directly exposing MemoryHandler's internal buffer of LogRecords
# for an API, as we can control the format and size directly.
api_log_buffer = deque(maxlen=MEMORY_LOG_CAPACITY)

class JsonFormatter(logging.Formatter):
    """
    Custom formatter to output logs in JSON format.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(), # Evaluate the message
            "loggerName": record.name,
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        # Add any extra fields passed to the logger
        standard_keys = set(log_entry.keys()).union({
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "levelno", "msecs", "msg", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName"
        })
        # Include extra fields passed to logger calls
        if hasattr(record, 'extra_fields') and isinstance(record.extra_fields, dict):
            for key, value in record.extra_fields.items():
                 if key not in standard_keys: # Avoid overriding standard fields
                    log_entry[key] = value
        elif record.args and isinstance(record.args, dict) and not (len(record.args) == 1 and record.args[0] is None):
            # Fallback for older style extra if not explicitly passed in extra_fields
            # This part might need adjustment based on how 'extra' is used in logger calls
            pass


        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)

class ApiBufferHandler(logging.Handler):
    """
    A custom handler to push formatted log messages to our api_log_buffer deque.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        # For the API buffer, let's use a simple text format by default.
        # This can be changed to JsonFormatter if JSON strings are preferred in the API.
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S.%fZ' # ISO8601 format
        ))


    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            api_log_buffer.append(msg)
        except Exception:
            self.handleError(record)

def setup_application_logging() -> None:
    """
    Configures logging for the application.
    """
    numeric_log_level = getattr(logging, LOG_LEVEL_ENV, logging.INFO)

    # Main console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_log_level)

    if LOG_FORMAT_ENV == "json":
        main_formatter = JsonFormatter()
    else:
        main_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
        )
    console_handler.setFormatter(main_formatter)

    # Custom handler for the API log buffer
    api_buffer_handler = ApiBufferHandler()
    api_buffer_handler.setLevel(numeric_log_level) # Capture logs at the configured level

    app_logger = logging.getLogger(APP_LOGGER_NAME)
    # Clear existing handlers on reload to avoid duplicates
    if app_logger.hasHandlers():
        app_logger.handlers.clear()

    app_logger.addHandler(console_handler) # Main handler for stdout
    app_logger.addHandler(api_buffer_handler) # Add handler for API buffer
    app_logger.setLevel(numeric_log_level)
    app_logger.propagate = False


    # Configure Uvicorn loggers to also use our handlers
    # This ensures Uvicorn's own operational logs (startup, errors) can also be captured
    for uvicorn_logger_name in ["uvicorn", "uvicorn.error"]: # uvicorn.access can be too noisy for API buffer
        uvicorn_logger = logging.getLogger(uvicorn_logger_name)
        if uvicorn_logger.hasHandlers():
            uvicorn_logger.handlers.clear()
        uvicorn_logger.addHandler(console_handler)
        uvicorn_logger.addHandler(api_buffer_handler) # Add API buffer handler here too
        uvicorn_logger.setLevel(numeric_log_level)
        uvicorn_logger.propagate = False # Avoid double logging if root logger is configured

    # Handle uvicorn.access separately if you want its format to be different
    # or if you don't want it in the api_log_buffer
    access_log = logging.getLogger("uvicorn.access")
    if access_log.hasHandlers():
        access_log.handlers.clear()
    access_log.addHandler(console_handler) # Only to console, not API buffer by default
    access_log.setLevel(numeric_log_level) # Or a different level like INFO
    access_log.propagate = False


    # Set levels for other verbose libraries
    logging.getLogger("aiohttp").setLevel(os.getenv("AIOHTTP_LOG_LEVEL", "WARNING").upper())
    logging.getLogger("websockets").setLevel(os.getenv("WEBSOCKETS_LOG_LEVEL", "WARNING").upper())
    logging.getLogger("faster_whisper").setLevel(os.getenv("FASTER_WHISPER_LOG_LEVEL", "WARNING").upper())
    logging.getLogger("librosa").setLevel(os.getenv("LIBROSA_LOG_LEVEL", "WARNING").upper())

def get_logger(name: str = None) -> logging.Logger:
    """
    Returns a logger instance.
    If 'name' is provided, it gets a child logger of the main application logger.
    Otherwise, it returns the main application logger.
    """
    if name:
        # Construct a child logger name based on the APP_LOGGER_NAME
        base_name_parts = APP_LOGGER_NAME.split('.')
        child_name_parts = name.split('.')
        
        if child_name_parts[:len(base_name_parts)] == base_name_parts:
            logger_name = name
        else:
            # Use the last part for brevity, prefixed by the app logger name
            logger_name = f"{APP_LOGGER_NAME}.{child_name_parts[-1]}" 
        return logging.getLogger(logger_name)
    return logging.getLogger(APP_LOGGER_NAME)

def get_api_logs(count: int = 0) -> list[str]:
    """
    Retrieves log messages from the api_log_buffer.
    If count is 0 or greater than buffer size, returns all logs in the buffer.
    Otherwise, returns the last 'count' logs.
    """
    if count > 0 and count < len(api_log_buffer):
        # Get the last 'count' elements
        return list(api_log_buffer)[-count:]
    return list(api_log_buffer) # Return all logs