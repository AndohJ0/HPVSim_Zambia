#!/bin/bash
#SBATCH --job-name=calibration_job
#SBATCH --output=output_calib.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=32

#source /storage/homefs/ja22x644/.local/lib/python3.10
python /storage/homefs/ja22x644/HPVSim_zambia/run_calibration.py
