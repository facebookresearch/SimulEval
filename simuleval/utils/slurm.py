# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import re
import subprocess
import sys
from argparse import ArgumentParser
from typing import Dict, List, Optional

from simuleval import options
from simuleval.utils.agent import get_agent_class
from simuleval.utils.arguments import cli_argument_list

logger = logging.getLogger("simuleval.slurm")


def mkdir_output_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except BaseException as be:
        logger.error(f"Failed to write results to {path}.")
        logger.error(be)
        logger.error("Skip writing predictions.")
        return False


def submit_slurm_job(
    config_dict: Optional[Dict] = None, parser: Optional[ArgumentParser] = None
) -> None:
    if config_dict is not None and "slurm" in config_dict:
        raise RuntimeError("--slurm is only available as a CLI argument")

    sweep_options = [
        [[key, v] for v in value]
        for key, value in config_dict.items()
        if isinstance(value, list)
    ]
    sweep_config_dict_list = []
    if len(sweep_options) > 0:
        for option_list in itertools.product(*sweep_options):
            sweep_config_dict_list.append({k: v for k, v in option_list})

        for x in sweep_options:
            if x[0][0] in config_dict:
                del config_dict[x[0][0]]

    cli_arguments = cli_argument_list(config_dict)
    parser = options.general_parser(config_dict, parser)
    options.add_evaluator_args(parser)
    options.add_scorer_args(parser, cli_arguments)
    options.add_slurm_args(parser)
    options.add_dataloader_args(parser, cli_arguments)
    system_class = get_agent_class(config_dict)
    system_class.add_args(parser)
    args = parser.parse_args(cli_argument_list(config_dict))
    args.output = os.path.abspath(args.output)
    assert mkdir_output_dir(args.output)

    if args.agent is None:
        args.agent = sys.argv[0]

    os.system(f"cp {args.agent} {args.output}/agent.py")
    _args = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if str(arg).isdigit() or str(arg).startswith("--"):
            _args.append(arg)
        else:
            _args.append(f'"{arg}"')
    command = " ".join(_args).strip()
    command = re.sub(r"(--slurm\S*(\s+[^-]\S+)*)", "", command).strip()
    if subprocess.check_output(["which", "simuleval"]).decode().strip() in command:
        command = re.sub(
            r"--agent\s+\S+", f"--agent {args.output}/agent.py", command
        ).strip()
    else:
        # Attention: not fully tested!
        command = re.sub(
            r"[^\"'\s]+\.py", f"{os.path.abspath(args.output)}/agent.py", command
        ).strip()

    sweep_command = ""
    sbatch_job_array_head = ""
    job_array_configs = ""

    if len(sweep_config_dict_list) > 0:
        job_array_configs = "declare -A JobArrayConfigs\n"
        for i, sub_config_dict in enumerate(sweep_config_dict_list):
            sub_config_string = " ".join(
                [f"--{k.replace('_', '-')} {v}" for k, v in sub_config_dict.items()]
            )
            job_array_configs += f'JobArrayConfigs[{i}]="{sub_config_string}"\n'

        job_array_configs += "\ndeclare -A JobArrayString\n"
        for i, sub_config_dict in enumerate(sweep_config_dict_list):
            sub_config_string = ".".join([str(v) for k, v in sub_config_dict.items()])
            job_array_configs += f'JobArrayString[{i}]="{sub_config_string}"\n'

        sweep_command = "${JobArrayConfigs[$SLURM_ARRAY_TASK_ID]}"
        sbatch_job_array_head = f"#SBATCH --array=0-{len(sweep_config_dict_list) - 1}"
        output_dir = (
            f"{args.output}" + "/results/${JobArrayString[$SLURM_ARRAY_TASK_ID]}"
        )
        log_path = f"{args.output}/logs/slurm-%A_%a.log"

    else:
        output_dir = args.output
        log_path = f"{args.output}/slurm-%j.log"

    if "--output" in command:
        command = re.sub(r"--output\s+\S+", f"--output {output_dir}", command).strip()
    else:
        command += f" --output {output_dir}"

    command = command.replace("--", "\\\n\t--")
    script = f"""#!/bin/bash
#SBATCH --time={args.slurm_time}
#SBATCH --partition={args.slurm_partition}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output="{log_path}"
#SBATCH --job-name="{args.slurm_job_name}"
{sbatch_job_array_head}

{job_array_configs}

mkdir -p {args.output}/logs
cd {os.path.abspath(args.output)}

GPU_ID=$SLURM_LOCALID

# Change to local a gpu id for debugging, e.g.
# GPU_ID=0


CUDA_VISIBLE_DEVICES=$GPU_ID {command} {sweep_command}
"""
    script_file = os.path.join(args.output, "script.sh")
    with open(script_file, "w") as f:
        f.writelines(script)

    process = subprocess.Popen(
        ["sbatch", script_file],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    logger.info("Using slurm.")
    logger.info(f"sbatch stdout: {stdout.decode('utf-8').strip()}")
    stderr = stderr.decode("utf-8").strip()
    if len(stderr) > 0:
        logger.info(f"sbatch stderr: {stderr.strip()}")
