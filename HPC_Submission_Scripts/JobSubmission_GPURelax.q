#!/bin/bash

#SBATCH -A p32212        # which account to debit hours from
#SBATCH --job-name="Relax"   # job name
#SBATCH -o myjob.0%j    #output and error file name (%j expands to jobI)
#SBATCH -e myjob.e%j    # output and error file name (%j expands to jobI)

#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1

#SBATCH --partition gengpu        # queue (partition) -- normal, development, ect.
#SBATCH -t 04:00:00      # time

#------------------------------------------------------------------------------
module purge all
module use /hpc/software/spack_v17d2/spack/share/modules/linux-rhel17-x86_64/
module load vasp/6.4.3-nvhpc-gcc-gpu-only
module load java/jdk1.8.0_191
module load numpy/1.19.2
module load openblas/0.3.21-gcc-4.8.5

#module load python-anaconda3
source /home/${USER}/.bashrc
source activate atomistic
#conda activate phonyop
/home/ysx6266/.conda/envs/atomistic/bin/python Relax_ASE.py

