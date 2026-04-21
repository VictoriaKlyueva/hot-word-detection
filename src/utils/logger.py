import logging
from pathlib import Path


class Logger:
    """Singleton logger with file and console handlers."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        log_dir = Path("data")
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("hot_word_detection")
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_dir / "log_file.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
    
    def info(self, msg: str) -> None:
        self._logger.info(msg)
    
    def error(self, msg: str) -> None:
        self._logger.error(msg)
    
    def warning(self, msg: str) -> None:
        self._logger.warning(msg)
    
    def debug(self, msg: str) -> None:
        self._logger.debug(msg)


logger = Logger()