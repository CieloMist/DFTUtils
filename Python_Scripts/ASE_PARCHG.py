# ----------------------------------------------------------------------- #
# Imports
    # System
import os
import sys
import json
import shutil as sh

# ASE
from ase.calculators.vasp import Vasp

    # File io
from ase.io import read, write
from asekpd import safe_kgrid_from_cell_volume

    # mine
from DFTUtils import make_directories_from_list
# ----------------------------------- #
# Read in VASP settings
os.environ["VASP_PP_PATH"] = "/projects/b1027/Pseudopotentials.64"
os.environ["VASP_COMMAND"] = "mpirun -n $SLURM_NTASKS vasp_std"

    # Read Input Settings - all INCAR and POTCAR settings specified here
with open('vasp_settings.json') as json_file:
    vasp_settings = json.load(json_file)
    json_file.close()

kpd = vasp_settings.pop('kpd')

# parchg settings
with open('parchg_settings.json') as json_file:
    parchg_settings = json.load(json_file)
    json_file.close()

directory_suffix = parchg_settings.pop('directory_suffix', None)
run_dft = parchg_settings.pop('run_dft', True)
# ----------------------------------- #
# Parchg directory
dirlist = ['PARCHG' if directory_suffix == None else 'PARCHG_' + directory_suffix]
make_directories_from_list(dirlist, delete = True)
os.chdir(dirlist[0])

# ----------------------------------------------------------------------- #
# Calculation Details
    # Set Initial Structure
struct = read('../Initial.traj', format = 'traj')

    # KPOINTS
kpts_list = safe_kgrid_from_cell_volume(struct, kpd)

# ----------------------------------- #
# Run dft if desired
if run_dft:

    calc = Vasp(**vasp_settings, kpts = kpts_list)
    struct.calc = calc

    struct.get_potential_energy()

# ----------------------------------- #
# Otherwise just run PARCHG postprocessing
vasp_settings['lpard'] = True
for setting in parchg_settings.keys():
    vasp_settings[setting] = parchg_settings[setting]

calc = Vasp(**vasp_settings, kpts = kpts_list)
struct.calc = calc
struct.get_potential_energy()