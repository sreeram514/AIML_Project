import logging
import os

logger = None


def initiate_logger(identifier, log_filename, level="DEBUG", log_dir="/var/tmp"):
    """
    Initializes loggers and creates log file under given log_dir
    :param identifier: Unique identifier - ex: pit-validaiton, pit-functional
    :param log_filename: Name of the log file
    :param level: Logging level, default: DEBUG
    :param log_dir: Directory to store log file, default: /var/tmp
    :return:
    """

    global logger

    logger = logging.getLogger(identifier)
    formatter = logging.Formatter(f'%(asctime)s - {identifier.upper()}-%(levelname)-5s - %(message)s')

    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    except Exception as e:
        raise Exception(f"ERROR: Unable to create log directory {e}")
    log_path = os.path.join(log_dir, log_filename)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    fh = logging.FileHandler(log_path)
    lvl_to_val = {"DEBUG": logging.DEBUG,
                  "INFO": logging.INFO,
                  "WARNING": logging.WARNING,
                  "ERROR": logging.ERROR,
                  "CRITICAL": logging.CRITICAL}
    fh.setFormatter(formatter)
    logger.setLevel(lvl_to_val[level])
    logger.addHandler(fh)
    logger.debug(f"Logs will be stored in {log_path}")


def start_end_log(stage):
    global logger

    def add_log(*args, **kwargs):
        stage_name = " ".join(stage.__name__.split("_")).title()
        logger.info(f"STARTED: {stage_name} stage")
        stage(*args, **kwargs)
        logger.info(f"COMPLETED: {stage_name} stage")
    return add_log
