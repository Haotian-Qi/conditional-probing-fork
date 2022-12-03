import argparse
import datetime as dt
import os
import re
from typing import Tuple

# Obtains cluster name via environment
CLUSTER = "bessemer" if os.environ.get("SGE_CLUSTER_NAME") is None else "sharc"

# Directories
REPORTS_DIR = "reports"
CONFIGS_DIR = "configs"
SCRIPTS_DIR = "scripts"
CURRENT_FILE_DIR = os.path.abspath(os.path.dirname(__file__))

# Path templates
JOB_SCRIPT_TEMPLATE = os.path.join(
    SCRIPTS_DIR,
    f"{CLUSTER}.sh.template",
)
JOB_COMMAND_TEMPLATE = "python3 vinfo/experiment.py {config}"

# Regexes
DATASET_PATH_REGEXES = {
    dataset: rf"(?<=\&id{dataset}path )(.*)" for dataset in ("dev", "train", "test")
}
REPORT_PATH_REGEX = r"(?<=\&id_reporting_root )(.*)"
TASK_NAME_REGEX = r"(?<=task_name: )(.*)"

# Paths to datasets
DATASET_PATHS = {
    "tsv": {
        dataset: os.path.join(os.getcwd(), f"data/sst2-{dataset}.tsv")
        for dataset in ("dev", "train", "test")
    },
    "conll": {
        dataset: os.path.join(os.getcwd(), f"data/{dataset}.ontonotes.withdep.conll")
        for dataset in ("dev", "train", "test")
    },
}


def get_script_name(config_filename: str) -> str:
    """
    Returns a filename for a job script file based on the config file name
    """
    return config_filename.rsplit(".", maxsplit=1)[0] + ".sh"


def get_task_name(config: str) -> str:
    match = re.search(TASK_NAME_REGEX, config)
    if match:
        return match.group().lower()
    return None


def write_config_file(path: str) -> Tuple[str, str]:
    """
    Writes a config `.yaml` file in preparation for HPC job submission.

    Returns a tuple of the path where the report will be written,
    and the path to the config file generated.
    """
    abs_config_dir = os.path.join(CURRENT_FILE_DIR, CONFIGS_DIR)
    if not os.path.exists(abs_config_dir):
        os.makedirs(abs_config_dir)

    with open(path, "r") as f:
        config = f.read()

    task_name = get_task_name(config)
    if task_name in ("sst",):
        file_type = "tsv"
    else:
        file_type = "conll"

    for dataset in DATASET_PATH_REGEXES:
        config = re.sub(
            DATASET_PATH_REGEXES[dataset],
            DATASET_PATHS[file_type][dataset],
            config,
            count=1,
        )

    config_filename = os.path.basename(path)
    reporting_root = os.path.join(
        CURRENT_FILE_DIR, REPORTS_DIR, f"{config_filename}.results"
    )
    config = re.sub(REPORT_PATH_REGEX, reporting_root, config, count=1)

    new_config_file_path = os.path.join(abs_config_dir, config_filename)
    with open(new_config_file_path, "w") as f:
        f.write(config)

    return reporting_root, new_config_file_path


def write_submission_script(email: str, config_file_path: str) -> str:
    """
    Writes a new submission script for executing the experiment
    with the given config file path.

    Returns the path to the new config file.
    """
    with open(os.path.join(CURRENT_FILE_DIR, JOB_SCRIPT_TEMPLATE), "r") as f:
        script_content = f.read()

    script_content = script_content.replace("<<email>>", email)
    script_content = script_content.replace(
        "<<main>>", JOB_COMMAND_TEMPLATE.format(config=config_file_path)
    )

    abs_scripts_dir = os.path.join(CURRENT_FILE_DIR, SCRIPTS_DIR)
    if not os.path.exists(abs_scripts_dir):
        os.makedirs(abs_scripts_dir)

    config_filename = os.path.basename(config_file_path)
    new_script_path = os.path.join(abs_scripts_dir, get_script_name(config_filename))
    with open(new_script_path, "w") as f:
        f.write(script_content)

    return new_script_path


def submit_job(script_path: str):
    if CLUSTER == "sharc":
        os.system(f"qsub {script_path}")
    else:
        os.system(f"sbatch {script_path}")


def parse_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="The path to the config .yaml file to use.")
    parser.add_argument("email", help="The email address to send HPC updates to.")

    return parser, parser.parse_args()


def main() -> None:
    parser, args = parse_args()

    reporting_root, config_file_path = write_config_file(args.config)
    print(f"Report will be written to {reporting_root}")
    print(f"New .yaml file written to {config_file_path}")

    submission_script_path = write_submission_script(args.email, config_file_path)
    print(f"New submission file written {submission_script_path}")

    submit_job(submission_script_path)


if __name__ == "__main__":
    main()
