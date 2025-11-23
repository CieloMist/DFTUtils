#!/bin/bash

#SBATCH -A p32212        # which account to debit hours from
#SBATCH --job-name="DSTRESSSUB"   # job name
#SBATCH -o myjob.0%j    #output and error file name (%j expands to jobI)
#SBATCH -e myjob.e%j    # output and error file name (%j expands to jobI)

#SBATCH -N 1                    # number of nodes to use
#SBATCH --ntasks-per-node=40    # num processors per node
#SBATCH -p short        # queue (partition) -- normal, development, ect.
#SBATCH -t 00:30:00      # time

#------------------------------------------------------------------------------
module purge all
module use /hpc/software/spack_v17d2/spack/share/modules/linux-rhel17-x86_64/
module load vasp/6.4.2-openmpi-intel-hdf5-cpu-only
module load java/jdk1.8.0_191
module load numpy/1.19.2
module load openblas/0.3.21-gcc-4.8.5

/home/ysx6266/.conda/envs/phonyop/bin/python Lobster.py