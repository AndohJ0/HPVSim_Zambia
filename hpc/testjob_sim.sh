#!/bin/bash
#SBATCH --job-name=my_python_job
#SBATCH --output=output_sim.txt
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16

source /storage/homefs/ja22x644/HPVSim_zambia/hpvsim_env/bin/activate
python /storage/homefs/ja22x644/HPVSim_zambia/run_sim.py
