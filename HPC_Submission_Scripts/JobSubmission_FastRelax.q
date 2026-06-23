#!/bin/bash

#SBATCH -A p32212        # which account to debit hours from
#SBATCH --job-name="FastRelax"   # job name
#SBATCH -o myjob.0%j    #output and error file name (%j expands to jobI)
#SBATCH -e myjob.e%j    # output and error file name (%j expands to jobI)

#SBATCH -p short
#SBATCH -N 1			# number of nodes to use
#SBATCH --ntasks-per-node=64	# num processors per node
#SBATCH --mem-per-cpu=4G        # Memory

#SBATCH --constraint="[quest10|quest11|quest12|quest13]"

#SBATCH -p short        # queue (partition) -- normal, development, ect.
#SBATCH -t 04:00:00      # time

#------------------------------------------------------------------------------
module purge all

#module load python-anaconda3
source /home/${USER}/.bashrc
source activate atomistic
#conda activate phonyop
/home/ysx6266/.conda/envs/atomistic/bin/python Fast_Relax_ASE.py

