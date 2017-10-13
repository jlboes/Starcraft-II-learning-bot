import logging
from logging.config import fileConfig

fileConfig('logging_config.ini')


class ScLogger:
    logger = logger = logging.getLogger()

    @staticmethod
    def log(message):
        ScLogger.logger.info("[SC LOGGER] : %s", message)

    @staticmethod
    def logbo(message):
        ScLogger.logger.info("[SC LOGGER][BUILD ORDER] : %s", message)

    @staticmethod
    def logReward(message):
        ScLogger.logger.info("[SC LOGGER][REWARD] : %.2f", message)
