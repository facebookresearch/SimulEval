# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys
import logging
import subprocess
from simuleval import options

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


def submit_slurm_job(args=None) -> None:
    if args is None:
        parser = options.general_parser()
        options.add_evaluator_args(parser)
        options.add_scorer_args(parser)
        options.add_slurm_args(parser)
        args, _ = parser.parse_known_args()

    args.output = os.path.abspath(args.output)
    assert mkdir_output_dir(args.output)

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

    if "--output" in command:
        command = re.sub(r"--output\s+\S+", f"--output {args.output}", command).strip()
    else:
        command += f" --output {args.output}"

    command = command.replace("--", "\\\n\t--")
    script = f"""#!/bin/bash
#SBATCH --time={args.slurm_time}
#SBATCH --partition={args.slurm_partition}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output="{args.output}/slurm-%j.log"
#SBATCH --job-name="{args.slurm_job_name}"

cd {os.path.abspath(args.output)}

GPU_ID=$SLURM_LOCALID

# Change to local a gpu id for debugging, e.g.
# GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID {command}
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
        logger.info(f"sbatch stderr: {stderr.decode('utf-8').strip()}")
