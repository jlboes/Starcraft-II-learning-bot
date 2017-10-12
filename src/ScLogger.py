import logging
from logging.config import fileConfig

fileConfig('logging_config.ini')


class ScLogger:
    logger = logger = logging.getLogger()

    @staticmethod
    def logbo(message):
        ScLogger.logger.info("[BUILD ORDER] : %s",message)

    @staticmethod
    def logReward(message):
        ScLogger.logger.info("[REWARD] : %i",message)
