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

# ----------------------------------- #
# Read in VASP settings
os.environ["VASP_PP_PATH"] = "/projects/b1027/Pseudopotentials.64"
os.environ["VASP_COMMAND"] = "mpirun -n $SLURM_NTASKS vasp_std"

    # Read Input Settings - all INCAR and POTCAR settings specified here
with open('vasp_settings.json') as json_file:
    vasp_settings = json.load(json_file)
    json_file.close()

kpd = vasp_settings.pop('kpd')

# ----------------------------------------------------------------------- #
# Calculation Details
    # Set Initial Structure
struct = read('Initial.traj', format = 'traj')

    # KPOINTS
kpts_list = safe_kgrid_from_cell_volume(struct, kpd)

# ----------------------------------- #
# Set Calculator
calc = Vasp(**vasp_settings, kpts = kpts_list)
struct.calc = calc

struct.get_potential_energy()

write('Final.traj', struct, format = 'traj')

# ----------------------------------- #
# High-Res DOS calculation:
# struct_dos = struct.copy()
# vasp_settings['ismear'] = -5
# vasp_settings['lelf'] = True
# struct_dos.calc = Vasp(**vasp_settings, kpts = kpts_list)
# struct_dos.get_potential_energy()

