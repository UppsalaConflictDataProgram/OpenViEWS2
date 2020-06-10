""" Slurm python interface """
from typing import Any, Dict, List, Optional
import subprocess
import string
import os
import uuid
import logging
import datetime

from typing_extensions import Literal

from views.config import SLURM as SLURM_CFG
from views.config import DIR_STORAGE
from views.utils import io

log = logging.getLogger(__name__)

THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))

SLURM_USERNAME: str = SLURM_CFG["username"]
SLURM_PROJECT: str = SLURM_CFG["project"]


def _parse_jobinfo(jobinfo_str: str) -> Dict[str, Any]:
    """ Get a dict of jobinfos keyed by slurm job ids

    Args:
        jobinfo_str: Output from jobinfo as a string
    Returns:
        jobs: dict of dicts keyed by running/waiting then slurm_job_id

    """

    def get_lines_between_start_and_empty(
        lines: List[str], start: str, headers: List[str],
    ):
        """ Get list of lines of running jobs """

        lines_running = []
        started_running = False
        for line in lines:
            # If we find the header go to next line and
            if start in line:
                started_running = True
                continue

            if started_running:
                if line == "":
                    break
                # else
                lines_running.append(line)

        # Drop the headers line
        for line in lines_running:
            all_headers = True
            for header in headers:
                if header not in line:
                    all_headers = False
            if all_headers:
                lines_running.remove(line)

        return lines_running

    def parse_lines_into_jobs(lines, headers):
        """ Get dict of running jobs from their lines """

        jobs = {}

        for line in lines:
            # Fields are separated by whitespace
            fields = line.split()
            job = {}
            job_id = int(fields[0])
            for index, header in enumerate(headers):
                job[header] = fields[index]
            jobs[job_id] = job.copy()

        return jobs

    headers_running = [
        "JOBID",
        "PARTITION",
        "NAME",
        "USER",
        "ACCOUNT",
        "ST",
        "START_TIME",
        "TIME_LEFT",
        "NODES",
        "CPUS",
        "NODELIST(REASON)",
    ]

    # Actual output contains DEPENDENCY too but it's mostly empty so
    # we drop it
    headers_waiting = [
        "JOBID",
        "POS",
        "PARTITION",
        "NAME",
        "USER",
        "ACCOUNT",
        "ST",
        "START_TIME",
        "TIME_LEFT",
        "PRIORITY",
        "CPUS",
        "NODELIST(REASON)",
        "FEATURES",
    ]

    jobs_running = {}
    jobs_waiting = {}

    lines_running = get_lines_between_start_and_empty(
        jobinfo_str.splitlines(),
        start="Running jobs:",
        headers=headers_running,
    )
    jobs_running = parse_lines_into_jobs(lines_running, headers_running)

    lines_waiting = get_lines_between_start_and_empty(
        jobinfo_str.splitlines(),
        start="Waiting jobs:",
        headers=headers_waiting,
    )
    jobs_waiting = parse_lines_into_jobs(lines_waiting, headers_waiting)

    jobs = {}
    jobs["running"] = jobs_running
    jobs["waiting"] = jobs_waiting

    return jobs


def get_jobinfo() -> Dict[Any, Any]:
    """ Get dictionary of jobs from jobinfo """

    def get_jobinfo_raw(username):
        """ Get the results from jobinfo -u username """

        command = f"jobinfo -u {username}"
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            # stdout=subprocess.PIPE,
            capture_output=True,
            encoding="utf-8",
        )
        return result.stdout

    username = SLURM_USERNAME
    jobinfo_raw = get_jobinfo_raw(username)
    jobs = _parse_jobinfo(jobinfo_raw)
    return jobs


def _make_runfile(
    command: str, jobtype: Literal["node", "core"], cores: int, hours: int
) -> str:

    # pylint: disable=too-many-arguments
    def _make_runfile_str(
        command: str,
        project: str,
        jobtype: str,
        cores: int,
        time: str,
        name: str,
    ):
        """ Create a slurm runfile string

        Args:
            project: slurm project id
            jobtype: "core" or "node"
            cores: number of cores
            time: time like "8:00:00" for 8 hours
            name: job name, make it unique
            command: the command to run
        Returns:
            runfile: A string of a slurm runfile

        """

        path_template = os.path.join(
            THIS_DIR, "templates", f"runfile_{jobtype}.txt"
        )
        with open(path_template, "r") as f:
            template_str = f.read()

        template = string.Template(template_str)

        dir_logs = os.path.join(DIR_STORAGE, "logs", "slurm")
        io.create_directory(dir_logs)
        log_location = os.path.join(dir_logs, f"{name}.log")

        msg = "jobtype must be core or node!"
        if jobtype not in ["core", "node"]:
            raise TypeError(msg)

        if jobtype == "core":
            mapping = {
                "PROJECT_ID": project,
                "JOBTYPE": jobtype,
                "N_CORES": cores,
                "TIME": time,
                "NAME": name,
                "LOGFILE_LOCATION": log_location,
                "COMMAND": command,
            }
        # Don't have N_CORES for node jobs.
        elif jobtype == "node":
            mapping = {
                "PROJECT_ID": project,
                "JOBTYPE": jobtype,
                "TIME": time,
                "NAME": name,
                "LOGFILE_LOCATION": log_location,
                "COMMAND": command,
            }

        runfile = template.substitute(mapping)

        return runfile

    def make_path_runfile(name: str):
        """ Make a runfile in DIR_STORAGE/runfiles/{name}.sh, creating dir """
        dir_runfiles = os.path.join(DIR_STORAGE, "runfiles")
        io.create_directory(dir_runfiles)
        return os.path.join(dir_runfiles, f"{name}.sh")

    def make_job_name():
        name_id = str(uuid.uuid4()).split("-")[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"job_{timestamp}_{name_id}"
        return name

    name = make_job_name()
    path_runfile = make_path_runfile(name)
    time_str = f"{str(hours)}:00:00"

    runfile_str = _make_runfile_str(
        command=command,
        project=SLURM_PROJECT,
        jobtype=jobtype,
        cores=cores,
        time=time_str,
        name=name,
    )

    with open(path_runfile, "w") as f:
        f.write(runfile_str)
    log.info(f"Wrote runfile to {path_runfile}")

    return path_runfile


def _submit_runfile(path: str, clusters: List[str]) -> Optional[int]:
    """ Submit a runfile to slurm using the sbatch command

    Args:
        path: path to runfile to submit
        clusters: list of clusters to submit the job to. Slurm will send
                  the job to the cluster that will start it first.
    Returns:
        job_id: integer job_id or None if slurm "failed"
    """

    clusters_str = ",".join(clusters)
    sbatch_command = f"sbatch --clusters={clusters_str} {path}"
    try:
        result = subprocess.run(
            sbatch_command,
            shell=True,
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
        response = result.stdout
    except subprocess.CalledProcessError:
        # Sometimes sbatch times out but still submits the job
        response = ""
        log.warning(f"sbatch might have failed on cmd: {sbatch_command}")

    success_str = "Submitted batch job "
    job_id: Optional[int]
    if success_str in response:
        job_id = [int(s) for s in response.split() if s.isdigit()][0]
        log.info(f"Submitted job to job_id: {job_id}")
    else:
        log.warning("Slurm unhappy")
        job_id = None

    return job_id


def run_command(
    command: str,
    hours: int = 24,
    cores: int = 20,
    jobtype: Literal["node", "core"] = "node",
    clusters: Optional[List[str]] = None,
) -> Optional[int]:
    """ Submit a job to run a command """

    if not isinstance(clusters, list):
        clusters = ["rackham", "snowy"]

    path = _make_runfile(command, cores=cores, jobtype=jobtype, hours=hours)
    job_id = _submit_runfile(path, clusters)

    return job_id
