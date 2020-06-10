""" Config module. Reads config.yaml in repo root and exposes vars """
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import os
import copy
import json
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGFMT = "[%(asctime)s] - %(name)s:%(lineno)d - %(levelname)s - %(message)s"


def _resolve(path: str) -> str:
    """ Resolve env vars and home in path """
    return os.path.expanduser(os.path.expandvars(path))


# pylint: disable=too-many-instance-attributes
@dataclass
class Db:
    """ Holds connection options for connecting through sqlalchemy """

    user: str
    host: str
    dbname: str
    port: int
    password: Optional[str] = None
    use_ssl: Optional[bool] = False
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_rootcert: Optional[str] = None

    @property
    def connectstring(self) -> str:
        """ Get a connectstring """

        if self.password:
            userpart = f"{self.user}:{self.password}"
        else:
            userpart = self.user

        return f"postgresql://{userpart}@{self.host}:{self.port}/{self.dbname}"

    @property
    def connect_args(self) -> Dict[str, str]:
        """ Get dict of connect_args """

        if self.use_ssl:
            assert self.ssl_cert
            assert self.ssl_key
            assert self.ssl_rootcert
            connectargs = {
                "sslmode": "require",
                "sslcert": _resolve(self.ssl_cert),
                "sslkey": _resolve(self.ssl_key),
                "sslrootcert": _resolve(self.ssl_rootcert),
            }
        else:
            connectargs = dict()

        return connectargs

    def __repr__(self):
        repdict = copy.copy(self.__dict__)

        # Never log the password
        if self.password:
            repdict["password"] = "******"
            repdict["connectstring"] = self.connectstring.replace(
                self.password, "******"
            )

        return json.dumps(repdict)

    def __str__(self):
        return self.__repr__()


def _get_configfile() -> Dict[str, Any]:
    """ Read the raw configfile """
    with open(os.path.join(REPO_ROOT, "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def _get_dirs() -> Tuple[str, str]:
    """ Get and resolve all the directories in config.yaml """
    config = _get_configfile()

    dir_storage = config["dirs"]["storage"]
    dir_scratch = config["dirs"]["scratch"]

    if not dir_storage:
        dir_storage = os.path.join(REPO_ROOT, "storage")

    if not dir_scratch:
        dir_scratch = os.path.join(dir_storage, "scratch")

    dir_storage = _resolve(dir_storage)
    dir_scratch = _resolve(dir_scratch)

    return dir_storage, dir_scratch


def _get_databases() -> Dict[str, Db]:
    """ Get all the database configs in config.yaml """
    config = _get_configfile()

    dbs = dict()
    for db_name, db_spec in config["databases"].items():
        dbs[db_name] = Db(**db_spec)

    dbs["default"] = dbs[config["default_database"]]

    return dbs


def _get_slurm_cfg() -> Dict[str, str]:
    config = _get_configfile()
    if "slurm" in config.keys():
        slurm_cfg = config["slurm"]
    else:
        slurm_cfg = {"username": "", "project": ""}

    return slurm_cfg


DIR_STORAGE, DIR_SCRATCH = _get_dirs()
DATABASES = _get_databases()
SLURM = _get_slurm_cfg()
