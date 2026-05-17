import logging
import sys


def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers or logging.getLogger().handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(module)s.%(funcName)s(): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)

    return logger
