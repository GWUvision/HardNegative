#!/bin/bash
#SBATCH -N 1
#SBATCH -p large-gpu
#SBATCH -t 7-00:00:00 
#SBATCH --mail-type=begin 
#SBATCH --mail-type=end  
#SBATCH --mail-user=xuanhong@gwu.edu

srun python main.py --Data HOTEL --model R50 --dim 256 --lr 1e-1 --order 1 --g 4 --lam 0.1 --semi 1 --ep 200