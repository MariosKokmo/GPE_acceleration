from src.run import main as run_code

class Simulator():
    """
    This class provides a simple interface to run the simulation code.
    """
    def __init__(self, configFile):
        self.configFile = configFile
    
    def run(self):
        run_code(self.configFile)