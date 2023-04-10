#!/usr/bin/env python3

import argparse
import os
import re
from typing import Optional, Tuple

# HPC commands
CLUSTER = "bessemer" if os.environ.get("SGE_CLUSTER_NAME") is None else "sharc"
SUBMIT_COMMAND = "sbatch" if CLUSTER == "bessemer" else "qsub"

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

# Regexes and patterns
REPORT_PATH_REGEX = r"(?<=\&reporting_root )(.*)"
QOS_CONFIG_LINE = "#SBATCH --qos=gpu\n"


def get_script_name(config_filename: str) -> str:
    """
    Returns a filename for a job script file based on the config file name
    """
    return config_filename.rsplit(".", maxsplit=1)[0] + ".sh"


def write_config_file(
    input_path: str,
    output_config_filename: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Writes a config `.yaml` file in preparation for HPC job submission.

    Returns a tuple of the path where the report will be written,
    and the path to the config file generated.
    """
    abs_output_config_dir = os.path.join(CURRENT_FILE_DIR, CONFIGS_DIR)
    if not os.path.exists(abs_output_config_dir):
        os.makedirs(abs_output_config_dir)

    with open(input_path, "r") as f:
        config = f.read()

    input_config_filename = os.path.basename(input_path)
    if output_config_filename is None:
        output_config_filename = input_config_filename
    else:
        output_config_filename = output_config_filename.replace(
            "%c", input_config_filename.split(".", maxsplit=1)[0]
        )
    report_dirname = f"{output_config_filename}.results"

    reporting_root = os.path.join(CURRENT_FILE_DIR, REPORTS_DIR, report_dirname)
    config = re.sub(REPORT_PATH_REGEX, reporting_root, config, count=1)

    new_config_file_path = os.path.join(abs_output_config_dir, output_config_filename)
    with open(new_config_file_path, "w+") as f:
        f.write(config)

    return reporting_root, new_config_file_path


def write_submission_script(
    email: str,
    config_file_path: str,
    use_dcs_gpu: bool,
) -> str:
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
    if use_dcs_gpu:
        script_content = script_content.replace(QOS_CONFIG_LINE, "")
        script_content = script_content.replace("<<partition>>", "dcs-gpu")
        script_content = script_content.replace("<<account>>", "dcs-res")
    else:
        script_content = script_content.replace("<<partition>>", "gpu")
        script_content = script_content.replace("<<account>>", "free")

    abs_scripts_dir = os.path.join(CURRENT_FILE_DIR, SCRIPTS_DIR)
    if not os.path.exists(abs_scripts_dir):
        os.makedirs(abs_scripts_dir)

    config_filename = os.path.basename(config_file_path)
    new_script_path = os.path.join(abs_scripts_dir, get_script_name(config_filename))
    with open(new_script_path, "w") as f:
        f.write(script_content)

    return new_script_path


def get_email():
    email_path = os.path.join(CURRENT_FILE_DIR, "email")
    if os.path.exists(email_path):
        with open(email_path, "r") as f:
            return f.read().strip()
    return None


def submit_job(script_path: str):
    os.system(f"{SUBMIT_COMMAND} {script_path}")


def parse_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="The path to the config .yaml file to use.")
    parser.add_argument(
        "-e",
        "--email",
        help="The email address to send HPC updates to.",
        default=get_email(),
    )
    parser.add_argument(
        "-o",
        "--output-config",
        help=(
            "Override the output config filename. "
            "By default, the name is the same as the config template passed. "
            "%%c will be replaced with the name of the config file without extension."
        ),
    )
    parser.add_argument(
        "--use-dcs-gpu",
        help=(
            "Whether or not to use the DCS GPU nodes. "
            "Raises an error if not on Bessemer."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--config-only",
        help="If passed, then only outputs the .yaml config file.",
        action="store_true",
    )

    return parser, parser.parse_args()


def main() -> None:
    parser, args = parse_args()
    if args.email is None:
        parser.error(
            "Email address not found. Create 'email' file, or pass -e flag with email"
        )
    if args.use_dcs_gpu and CLUSTER != "bessemer":
        parser.error("DCS GPU nodes are only available on Bessemer")

    reporting_root, config_file_path = write_config_file(
        args.config, args.output_config
    )
    print(f"I: Results will be written to {reporting_root}")
    print(f"I: Config file written to {config_file_path}")

    if args.config_only:
        return

    submission_script_path = write_submission_script(
        args.email, config_file_path, args.use_dcs_gpu
    )
    print(f"I: New submission file written {submission_script_path}")
    print(f"{SUBMIT_COMMAND}: ", end="")
    submit_job(submission_script_path)


if __name__ == "__main__":
    main()
