import logging
import sys
import os

class Logger:
    def __init__(self, name='default_logger', level=logging.INFO, log_file=None):
        """
        Initialize a reusable logger instance.

        Args:
            name (str): Name of the logger.
            log_file (str): Optional file path for logging to a file.
            level (int): Logging level (e.g., logging.INFO).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Avoid duplicate handlers
        if not self.logger.handlers:
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Console handler
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            # Optional file handler
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger

    def enable_debug(self):
        self.logger.setLevel(logging.DEBUG)

    def enable_info(self):
        self.logger.setLevel(logging.INFO)