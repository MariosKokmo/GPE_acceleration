"""Application-wide Configuration"""
import logging
from datetime import datetime as dt
from src.utils.setup_simulations import get_application_config

class application:
    def __init__(self, appConfigFile="appConfig.json"):
        self.logfile = None # the filename of the logger
        self.appLogfile = None # the logger at App level
        self.configFile = None # the configuration file at the app level
        self.device = None
        self.write_velocity = False
        self.phase_imaging = False
        self.appConfigFile = appConfigFile
        self._logger = None # Store the logger instance
        self.initialise()

    def initialise(self):
        appConfig = get_application_config(self.appConfigFile)
        self.appLogfile = appConfig["logfile"]
        self.configFile = appConfig["configFile"]
        self.logfile = self.appLogfile
        self.write_velocity = appConfig["write_velocity"]
        self.phase_imaging = appConfig["phase_imaging"]
        
        # Initialize the main logger
        self._setup_logger(self.appLogfile)
    
    def _setup_logger(self, logfile):
        """Setup logging configuration."""
        # Create logger
        self._logger = logging.getLogger('GPE_Simulation')
        self._logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        if self._logger.handlers:
            self._logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(logfile, mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(levelname)s]: %(asctime)s -- %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)
    
    @property
    def logger(self):
        """Get the logger instance."""
        return self._logger
    
    def set_logger(self, logfile):
        """Change the log file for the logger."""
        self.logfile = logfile
        self._setup_logger(logfile)

    def reset_logger(self):
        """Reset logger to use the application-level log file."""
        self.logfile = self.appLogfile
        self._setup_logger(self.appLogfile)
    
    def set_device(self, device):
        self.device = device

    def open_logger(self):
        """Deprecated: Returns logger for backward compatibility."""
        return self._logger
    
    def close_logger(self):
        """Deprecated: No-op for backward compatibility."""
        pass

    def time(self):
        """Deprecated: Use logger timestamps instead."""
        return dt.now()
