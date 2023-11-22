"""Application-wide Configuration"""
from datetime import datetime as dt

class application:
    def __init__(self, device):
        self.logger = None
        self.device = None
        self.time = dt.now()

    def set_logger(self, logger):
        self.logger = logger
    
    def set_device(self, device):
        self.device = device