""" Set global config options from environment vars """

__all__ = ["LOGFMT", "DB_CONNECTSTRING", "DIR_STORAGE"]

import os

LOGFMT = "[%(asctime)s] - %(name)s:%(lineno)d - %(levelname)s - %(message)s"


def _get_env_var_if_exists(name_env: str, default: str = "") -> str:
    if name_env in os.environ.keys():
        env_value = os.environ[name_env]
    else:
        env_value = default

    return env_value


DB_CONNECTSTRING: str = _get_env_var_if_exists(
    name_env="VIEWS2_DB_CONNECTSTRING"
)

DIR_STORAGE: str = _get_env_var_if_exists(
    name_env="VIEWS2_STORAGE",
    default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "storage")
    ),
)
