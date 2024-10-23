#!/bin/bash
#SBATCH --job-name=bash          # avoid lightning auto-debug configuration
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=6      # shall >#GPU to avoid overtime thread distribution 
#SBATCH --cpus-per-task=4        # number of OpenMP threads per MPI process
#SBATCH --mem=192GB               
#SBATCH --gres=gpu:4             # number of GPUs
#SBATCH --time 47:59:59           # time limit (D-HH:MM:ss)

#########################
####### Configs #########
#########################
CONDA_ENV_NAME=py310
CONDA_HOME=$(expr match $CONDA_PREFIX '\(.*miniconda\)')
WORKDIR=$(pwd)

#########################
####### Env loader ######
#########################
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}
module load cuda/12.1.1


dt=$(date '+%d/%m/%Y-%H:%M:%S')
echo "[$0] >>> Starttime => ${dt}"

#########################
####### Routine #########
#########################
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo ">>> Exec: python slm/train.py $@"
sleep 1

python slm/train.py $@ 