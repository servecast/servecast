#!/bin/bash
# ServeCast's GPU flow as one Slurm job: steps 1 to 5 of the On your GPU guide, on one GPU of a compute node.
#
#   sbatch servecast.sbatch                          example-a100.yaml, written with the example manifest that ships
#                                                    with ServeCast unless the directory has one; calibration my-a100
#   sbatch servecast.sbatch MANIFEST CALIBRATION     your manifest's path, and the calibration name its
#                                                    calibration.use gives
#
# Activate the Python environment ServeCast is installed in before sbatch: the job inherits it. Set the partition,
# the account and the time limit of your cluster below (a line starting with ##SBATCH is ignored). The outputs go to
# the directory sbatch is run from: profiling/, prediction/ and measurement/, and this job's log.
#SBATCH --job-name=servecast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=servecast-%j.log
##SBATCH --partition=PARTITION
##SBATCH --account=ACCOUNT

set -euo pipefail
manifest="${1:-example-a100.yaml}"
calibration="${2:-my-a100}"

servecast doctor
# without a manifest of yours, the example, written as the guide writes it; an example-a100.yaml already here is kept
[ $# -gt 0 ] || [ -f "$manifest" ] || servecast manifest show example-a100 > "$manifest"
servecast profile calibration -c "$manifest" --output-dir profiling
servecast calibrate --data profiling --name "$calibration" --overwrite
servecast predict -c "$manifest" --output-dir prediction
servecast profile serving -c "$manifest" --output-dir measurement
servecast compare prediction measurement
