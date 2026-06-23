#!/bin/bash

#SBATCH -A p32212        # which account to debit hours from
#SBATCH --job-name="Relax"   # job name
#SBATCH -o myjob.0%j    #output and error file name (%j expands to jobI)
#SBATCH -e myjob.e%j    # output and error file name (%j expands to jobI)

#SBATCH -N 2			# number of nodes to use
#SBATCH --ntasks-per-node=48	# num processors per node
#SBATCH --mem-per-cpu=4G # Memory per node


#SBATCH --constraint="[quest10|quest11|quest12|quest13]"

#SBATCH -p normal        # queue (partition) -- normal, development, ect.
#SBATCH -t 48:00:00      # time

#------------------------------------------------------------------------------
module purge all
module load quantum-espresso/7.5-openmpi-gcc-13.3.0

#module load python-anaconda3
source /home/${USER}/.bashrc
source activate atomistic
#conda activate phonyop
/home/ysx6266/.conda/envs/atomistic/bin/python Relax_QE.py

