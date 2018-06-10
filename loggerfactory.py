import logging.config
import logging
import os.path
from singleton import Singleton
LOGGING_CONF=os.path.join(os.path.dirname(__file__),
                          "logging.ini")

class LoggerManager(object):
    __metaclass__ = Singleton

    _loggers = {}

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def getLogger(name=None):
        if not name:
            logging.config.fileConfig(LOGGING_CONF, False)
            return logging.getLogger()
        elif name not in LoggerManager._loggers.keys():
            #print LoggerManager._loggers.keys()
            logging.config.fileConfig(LOGGING_CONF, False)
            LoggerManager._loggers[name] = logging.getLogger(str(name))
        return LoggerManager._loggers[name]    


#log=LoggerManager().getLogger("Hello")
#log.setLevel(level=logging.DEBUG)
