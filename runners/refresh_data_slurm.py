""" Refresh data via slurm """

import os
import sys
import logging

from views.apps.slurm.slurm import run_command
from views.config import LOGFMT
from views.utils.log import get_log_path

logging.basicConfig(
    level=logging.DEBUG,
    format=LOGFMT,
    handlers=[
        logging.FileHandler(get_log_path(__file__)),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger(__name__)


def _build_cmd_refresh_data() -> str:
    """ Just get a shell command for starting refersh_data.py """

    path_runner = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "refresh_data.py"
    )
    path_exec = sys.executable
    cmd = f"{path_exec} {path_runner} --all"
    return cmd


def main() -> None:
    run_command(command=_build_cmd_refresh_data(), hours=24)


if __name__ == "__main__":
    main()
