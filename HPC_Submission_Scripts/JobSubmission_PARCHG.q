#!/bin/bash

#SBATCH -A p32212        # which account to debit hours from
#SBATCH --job-name="PARCHG"   # job name
#SBATCH -o myjob.0%j    #output and error file name (%j expands to jobI)
#SBATCH -e myjob.e%j    # output and error file name (%j expands to jobI)

#SBATCH -p short
#SBATCH -N 8			# number of nodes to use
#SBATCH --ntasks-per-node=8	# num processors per node
#SBATCH --mem-per-cpu=4G # Memory per node


#SBATCH --constraint="[quest10|quest11|quest12|quest13]"

#SBATCH -p short        # queue (partition) -- normal, development, ect.
#SBATCH -t 02:00:00      # time

#------------------------------------------------------------------------------
module purge all
module use /hpc/software/spack_v17d2/spack/share/modules/linux-rhel17-x86_64/
module load vasp/6.4.2-openmpi-intel-hdf5-cpu-only
module load java/jdk1.8.0_191
module load numpy/1.19.2
module load openblas/0.3.21-gcc-4.8.5

#module load python-anaconda3
source /home/${USER}/.bashrc
source activate atomistic
#conda activate phonyop
/home/ysx6266/.conda/envs/atomistic/bin/python ASE_PARCHG.py

