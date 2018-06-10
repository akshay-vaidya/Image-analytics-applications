import logging
import os.path
import json
from singleton import Singleton

APP_CONF=os.path.join(os.path.dirname(__file__),
                          "itagconfig.json")

class ConfigListener(object):
    __metaclass__ = Singleton

    _configInstaces = {}
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def getConfigObject(name="IA_CONFIG"):
        if name not in ConfigListener._configInstaces.keys(): 
            myConfigLoader = ConfigLoader() 
            myConfigLoader.loadConfig()
            ConfigListener._configInstaces[name] = myConfigLoader.getConfig()
        return ConfigListener._configInstaces[name]
        
        

class ConfigLoader:
    def __init__(self):
        self.config_data = {}

    def loadConfig(self):
        with open(APP_CONF) as json_data_file:
            self.config_data = json.load(json_data_file)

    def getConfig(self):
        return self.config_data
        

    







