# ----------------------------------------------------------------------- #
# Imports
    # System
import os
import sys
import json
import subprocess
    # System +
import numpy as np
# ASE Relaxation Stuff
from ase.calculators.vasp import Vasp
from ase.optimize import BFGS, BFGSLineSearch, GPMin, MDMin
    # Filters/Masks
from ase.filters import FrechetCellFilter
from ase.filters import StrainFilter
from ase.stressbox import stressbox
    # File io
from ase.io import read, write
from asekpd import safe_kgrid_from_cell_volume
# Pymatgen Stuff
from pymatgen.io.vasp import Kpoints

# ----------------------------------- #
# Read in VASP settings
os.environ["VASP_PP_PATH"] = "/projects/b1027/Pseudopotentials.64"
os.environ["VASP_COMMAND"] = "mpirun -n $SLURM_NTASKS vasp_std"

    # Read Input Settings - all INCAR and POTCAR settings specified here
with open('vasp_settings.json') as json_file:
    vasp_settings = json.load(json_file)
    json_file.close()

kpd = vasp_settings.pop('kpd')

# ----------------------------------- #
# Read in filter settings
with open('filter_settings.json') as json_file:
    filter_settings = json.load(json_file)
    json_file.close()

filter_type = filter_settings.pop('filter')
optimizer = filter_settings.pop('optimizer')
fmax = filter_settings.pop('fmax')
restart = filter_settings.pop('restart')

# cover stressbox case, passed as a sub dictionary to filter_settings
try:
    stressbox_settings = filter_settings.pop('stressbox_settings')
    stressbox_settings['express'] = np.array(stressbox_settings['express'])
    stressbox_settings['fixstrain'] = np.array(stressbox_settings['fixstrain'])
except:
    pass

# ----------------------------------------------------------------------- #
# Calculation Details
    # Set Initial Structure
struct = read('Initial.traj', format = 'traj')

if (restart == True) and (os.path.exists(os.getcwd() + '/relax.traj')):
    struct = read('relax.traj', format = 'traj')

    # KPOINTS
kpts_list = safe_kgrid_from_cell_volume(struct, kpd)

# ----------------------------------- #
# Set Calculator
calc = Vasp(**vasp_settings, kpts = kpts_list)
struct.calc = calc

# ----------------------------------- #
# Set cell filter to optimize cell positions too
if filter_type == 'Frechet':
    struct_opt = FrechetCellFilter(struct, **filter_settings)
elif filter_type == 'strain':
    struct_opt = StrainFilter(struct, **filter_settings)
elif filter_type == 'stressbox':
    struct_opt = stressbox(struct, ref_atom = struct.copy(), **stressbox_settings)
else:
    struct_opt = struct.copy() # set no filter if no filter is provided

# ---------------------------------- #
# Set Optimizer
if optimizer == 'BFGSLineSearch':
    relaxer = BFGSLineSearch(struct_opt, trajectory = 'relax.traj', restart='BFGSLS_hessian.json')
elif optimizer == 'GPMin':
    relaxer = GPMin(struct_opt, trajectory = 'relax.traj', restart='gaussian_process.json')
elif optimizer == 'MDMin':
    relaxer = MDMin(struct_opt, trajectory = 'relax.traj', restart = 'MDMin_restart.json')
else:
    relaxer = BFGS(struct_opt, trajectory = 'relax.traj', restart='BFGS_hessian.json')
    
    # Run Relaxation
relaxer.run(fmax = fmax)

# ----------------------------------- #
# Write out relevant parameters
write('POSCAR_relaxed', struct, format = 'vasp')
struct.calc.write_json('calc_state.json')

# ----------------------------------- #
# High-Res DOS calculation:
struct_dos = struct.copy()
vasp_settings['ismear'] = -5
vasp_settings['lelf'] = True
struct_dos.calc = Vasp(**vasp_settings, kpts = kpts_list)
struct_dos.get_potential_energy()

