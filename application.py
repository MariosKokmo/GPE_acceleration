"""Application-wide Configuration"""
from datetime import datetime as dt
from utils.setup_simulations import get_application_config

class application:
    def __init__(self):
        self.logfile = None # the filename of the logger
        self.logger = None # the logger handler
        self.appLogfile = None # the logger at App level
        self.device = None
        self.write_velocity = False
        self.phase_imaging = False
        self.initialise()

    def initialise(self):
        appConfig = get_application_config()
        self.appLogfile = appConfig["logfile"]
        self.set_logger(self.appLogfile)
        self.write_velocity = appConfig["write_velocity"]
        self.phase_imaging = appConfig["phase_imaging"]
        
    def set_logger(self, logfile):
        self.logfile = logfile

    def reset_logger(self):
        self.logfile = self.appLogfile
    
    def set_device(self, device):
        self.device = device

    def open_logger(self):
        self.logger = open(self.logfile, 'a')
        return self.logger
    
    def close_logger(self):
        self.logger.close()

    def time(self):
        return dt.now()
