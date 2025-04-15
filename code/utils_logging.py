
import logging
import sys
import threading

import logging
import sys
import threading

class Logger_all:
    """
    A robust logger class that duplicates output to a primary log file and can direct messages to additional log files.

    Features:
        - Replaces sys.stdout and sys.stderr to log print statements.
        - Supports multiple additional log files identified by unique identifiers.
        - Thread-safe operations.
        - Context manager support for automatic resource management.

    Usage:
        with Logger_all(primary_log_path, print_to_terminal=True) as logger:
            logger.add_log_file('trn', training_log_path)
            logger.add_log_file('val', validation_log_path)
            
            print("This will be logged to the primary log and printed to the terminal.")
            logger.write_to_additional("This is a training log message.", 'trn')
            logger.write_to_additional("This is a validation log message.", 'val')
    """

    def __init__(self, primary_log_path, print_to_terminal=True):
        self.lock = threading.Lock()
        self.print_to_terminal = print_to_terminal

        # Primary logger setup
        self.primary_logger = logging.getLogger('primary_logger')
        self.primary_logger.setLevel(logging.DEBUG)
        self.primary_logger.propagate = False  # Prevent messages from being passed to the root logger

        if not self.primary_logger.handlers:
            # Primary file handler
            fh = logging.FileHandler(primary_log_path)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.primary_logger.addHandler(fh)

            # Console handler
            if self.print_to_terminal:
                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.DEBUG)
                ch.setFormatter(formatter)
                self.primary_logger.addHandler(ch)

        # Dictionary to hold additional loggers
        self.additional_loggers = {}

    def add_log_file(self, identifier, path, level=logging.INFO):
        """
        Adds an additional log file.

        Args:
            identifier (str): Unique identifier for the additional log file.
            path (str): Path to the additional log file.
            level (int): Logging level (default: logging.INFO).

        Raises:
            ValueError: If the identifier already exists.
            IOError: If the file cannot be opened.
        """
        with self.lock:
            if identifier in self.additional_loggers:
                raise ValueError(f"Identifier '{identifier}' already exists.")

            # Create a new logger for the additional log file
            logger = logging.getLogger(f'additional_logger_{identifier}')
            logger.setLevel(level)
            logger.propagate = False  # Prevent messages from being passed to the root logger

            # File handler for the additional logger
            try:
                fh = logging.FileHandler(path)
                fh.setLevel(level)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                fh.setFormatter(formatter)
                logger.addHandler(fh)
                self.additional_loggers[identifier] = logger
            except Exception as e:
                raise IOError(f"Failed to open additional log file '{path}': {e}")

    def write_to_additional(self, identifier, message, level='INFO'):
        """
        Writes a message to an additional log file and the primary log.

        Args:
            message (str): The message to log.
            identifier (str): The identifier of the additional log file.
            level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').

        Raises:
            KeyError: If the identifier does not exist.
            ValueError: If the logging level is invalid.
        """
        with self.lock:
            if identifier not in self.additional_loggers:
                raise KeyError(f"Identifier '{identifier}' does not exist.")

            level = level.upper()

            # Log to primary logger
            if level == 'DEBUG':
                self.primary_logger.debug(message)
            elif level == 'INFO':
                self.primary_logger.info(message)
            elif level == 'WARNING':
                self.primary_logger.warning(message)
            elif level == 'ERROR':
                self.primary_logger.error(message)
            elif level == 'CRITICAL':
                self.primary_logger.critical(message)
            else:
                raise ValueError(f"Invalid logging level: {level}")

            # Log to additional logger
            additional_logger = self.additional_loggers[identifier]
            if level == 'DEBUG':
                additional_logger.debug(message)
            elif level == 'INFO':
                additional_logger.info(message)
            elif level == 'WARNING':
                additional_logger.warning(message)
            elif level == 'ERROR':
                additional_logger.error(message)
            elif level == 'CRITICAL':
                additional_logger.critical(message)
            else:
                raise ValueError(f"Invalid logging level: {level}")

    def write(self, message):
        """
        Writes a message to the primary log and optionally to the terminal.

        Args:
            message (str): The message to log.
        """
        with self.lock:
            message = message.rstrip('\n')  # Remove any trailing newline
            if message:
                self.primary_logger.info(message)

    ### try to enable printing newlines, but not very well designed ...
    # def write(self, message):
    #     with self.lock:
    #         # If the message consists only of newline characters (e.g., "\n" or "\n\n")
    #         if message.strip('\n') == "":
    #             newline_count = message.count('\n')
    #             for _ in range(newline_count):
    #                 self.primary_logger.info("")
    #         else:
    #             message = message.rstrip('\n')
    #             if message:
    #                 self.primary_logger.info(message)
                
    def flush(self):
        """
        Flushes all handlers.
        """
        with self.lock:
            for handler in self.primary_logger.handlers:
                handler.flush()
            for logger in self.additional_loggers.values():
                for handler in logger.handlers:
                    handler.flush()

    def close(self):
        """
        Closes all handlers associated with the loggers.
        """
        with self.lock:
            # Close primary logger handlers
            for handler in self.primary_logger.handlers[:]:
                handler.close()
                self.primary_logger.removeHandler(handler)

            # Close additional logger handlers
            for logger in self.additional_loggers.values():
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)

    def __enter__(self):
        """
        Enables use of the logger with the 'with' statement.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures all handlers are closed when exiting the 'with' block.
        """
        self.close()

    def isatty(self):
        # This allows the logger to be used as sys.stderr in environments
        # where a TTY check is performed (e.g., by wandb).
        return False

