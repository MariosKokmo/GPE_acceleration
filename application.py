"""Application-wide Configuration"""
from datetime import datetime as dt

class application:
    def __init__(self):
        self.logfile = None # the filename of the logger
        self.logger = None # the logger handler
        self.device = None
        self.write_velocity = False

    def set_logger(self, logfile):
        self.logfile = logfile
    
    def set_device(self, device):
        self.device = device

    def open_logger(self):
        self.logger = open(self.logfile, 'a')
        return self.logger
    
    def close_logger(self):
        self.logger.close()

    def time(self):
        return dt.now()
