""" Logging utils """
from functools import wraps
import datetime
import logging
import os
import time
import uuid

from views.config import DIR_STORAGE

log = logging.getLogger(__name__)


def utc_now() -> str:
    """ Get current UTC time """
    return datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")


def get_log_path(caller_path: str) -> str:
    """ Get unique and timestamped path to a logfile """
    name = os.path.basename(caller_path).replace(".py", "")
    # Hopefully unique filename with timestamp and part of a uuid
    fname = f"{name}_{utc_now()}_{str(uuid.uuid4()).split('-')[0]}.log"
    path = os.path.join(DIR_STORAGE, "logs", fname)
    print(f"Logging to {path}")
    return path


def logtime(func):
    """This decorator logs the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        log.debug("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

    return wrapper
