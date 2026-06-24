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
# MACE
from mace.calculators import mace_mp
# META
# from fairchem.core import pretrained_mlip, FAIRChemCalculator

# ----------------------------------- #
# Read in filter settings
with open('filter_settings.json') as json_file:
    filter_settings = json.load(json_file)
    json_file.close()

filter_type = filter_settings.pop('filter', 'Frechet')
optimizer = filter_settings.pop('optimizer', 'BFGS')
fmax = filter_settings.pop('fmax', 0.005)
restart = filter_settings.pop('restart', False)

# cover stressbox case, passed as a sub dictionary to filter_settings
try:
    stressbox_settings = filter_settings.pop('stressbox_settings')
    stressbox_settings['express'] = np.array(stressbox_settings['express'])
    stressbox_settings['fixstrain'] = np.array(stressbox_settings['fixstrain'])
except:
    pass

# --------------------- #
# Read MACE settings
with open('MLIP_settings.json') as json_file:
    MLIP_settings = json.load(json_file)
    json_file.close()

fmax_MLIP = MLIP_settings.pop('fmax')

# ----------------------------------------------------------------------- #
# Calculation Details
    # Set Initial Structure
struct = read('Initial.traj', format = 'traj')

if (restart == True) and (os.path.exists(os.getcwd + '/relax.traj')):
    struct = read('relax.traj', format = 'traj')

# ------------------------------------------------------------ #
# Set cell filter to optimize cell positions too
if filter_type == 'Frechet':
    struct_opt = FrechetCellFilter(struct, **filter_settings)
elif filter_type == 'strain':
    struct_opt = StrainFilter(struct, **filter_settings)
elif filter_type == 'stressbox':
    ref_struct = struct.copy()
    struct_opt = stressbox(struct, ref_atom = ref_struct, **stressbox_settings)
else:
    struct_opt = struct.copy() # set no filter if no filter is provided

# ---------------------- #
# Set Optimizer
if optimizer == 'BFGSLineSearch':
    restart_file = 'BFGSLineSearch_hessian.json'
    relaxer = BFGSLineSearch(struct_opt, trajectory = 'relax.traj', restart=restart_file)
elif optimizer == 'GPMin':
    restart_file = 'gaussian_process.json'
    relaxer = GPMin(struct_opt, trajectory = 'relax.traj', restart=restart_file)
elif optimizer == 'MDMin':
    restart_file = 'MDMin.json'
    relaxer = MDMin(struct_opt, trajectory = 'relax.traj', restart = restart_file)
else:
    restart_file='BFGS_hessian.json'
    relaxer = BFGS(struct_opt, trajectory = 'relax.traj', restart=restart_file)
    
# ------------------------------------------------------------- #
# MACE PRE-RELAXATION
# default to MACE
calc = mace_mp(**MLIP_settings)
struct.calc = calc

relaxer.run(fmax = fmax_MLIP)