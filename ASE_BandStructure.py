# ----------------------------------------------------------------------- #
# Imports
    # System
import os
import sys
import json

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

with open('band_settings.json') as json_file:
    band_settings = json.load(json_file)
    json_file.close()

directory_suffix = band_settings.pop('directory_suffix', None)
run_dft = band_settings.pop('run_dft', True)
# ----------------------------------- #
# Parchg directory
dirlist = ['Band_Structure' if directory_suffix == None else 'Band_Structure_' + directory_suffix]
make_directories_from_list(dirlist, delete = True)
os.chdir(dirlist[0])

# ----------------------------------------------------------------------- #
# Calculation Details
    # Set Initial Structure
struct = read('../Initial.traj', format = 'traj')

    # KPOINTS
kpts_list = safe_kgrid_from_cell_volume(struct, kpd)

# ----------------------------------- #
# Set Calculator
if run_dft:
    calc = Vasp(**vasp_settings, kpts = kpts_list)
    struct.calc = calc

    struct.get_potential_energy()

# ----------------------------------- #
# Band Structure Calculation:
vasp_settings['isym'] = 0
vasp_settings['icharg'] = 11
vasp_settings['reciprocal'] = True

from ase.dft.kpoints import BandPath
path = BandPath(path = band_settings['path'], cell = struct.cell, special_points = band_settings['special_points'])
path = path.interpolate(npoints = band_settings['npoints'])
linemode_kpts = path.kpts

calc = Vasp(**vasp_settings)
calc.set(kpts=linemode_kpts)
struct.calc = calc
struct.get_potential_energy()

