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


def submit_slurm_job() -> None:
    parser = options.general_parser()
    options.add_evaluator_args(parser)
    options.add_slurm_args(parser)
    args, _ = parser.parse_known_args()

    assert mkdir_output_dir(args.output)
    os.system(f"cp {args.agent} {args.output}/agent.py")
    _args = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if str(arg).isdigit() or str(arg).startswith("--"):
            _args.append(arg)
        else:
            _args.append(f'"{arg}"')
    command = " ".join(_args)
    command = re.sub(r"(--slurm\S*(\s+[^-]\S+)*)", "", command).strip()
    command = re.sub(
        r"--agent\s+\S+", f"--agent {args.output}/agent.py", command
    ).strip()
    command = command.replace("--", "\\\n\t--")
    script = f"""#!/bin/bash
#SBATCH --time={args.slurm_time}
#SBATCH --partition={args.slurm_partition}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output="{args.output}/slurm-%j.log"
#SBATCH --job-name="{args.slurm_job_name}"

cd {os.path.abspath(os.getcwd())}

CUDA_VISIBLE_DEVICES=$SLURM_LOCALID {command}
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
