"""Application-wide Configuration"""
from datetime import datetime as dt

class application:
    def __init__(self):
        self.logger = None
        self.device = None
        self.write_velocity = False

    def set_logger(self, logger):
        self.logger = logger
    
    def set_device(self, device):
        self.device = device

    def time(self):
        return dt.now()
