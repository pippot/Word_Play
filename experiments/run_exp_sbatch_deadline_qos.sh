#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=167G
#SBATCH --qos=deadline
#SBATCH --gres=gpu:a40:4
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=32
#SBATCH --job-name=normative
#SBATCH --output=exp_log_files/normative_%j.out
#SBATCH --error=exp_log_files/normative_%j.error

exp_config_path=$1
echo "exp_config_path: '$exp_config_path'"

source /h/andrei/.bashrc
cd /h/andrei/normative_agents/experiments
conda deactivate
conda activate normative_3.11

echo "Running experiment..."
python run_exp.py --exp_config_path=$exp_config_path --results_dir="/h/andrei/normative_agents/results"

echo "Done."
