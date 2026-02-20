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
from fairchem.core import pretrained_mlip, FAIRChemCalculator

# ----------------------------------- #
# VASP Path Definitions
os.environ["VASP_PP_PATH"] = "/projects/b1027/Pseudopotentials.64"
os.environ["VASP_COMMAND"] = "mpirun -n $SLURM_NTASKS vasp_std"

# ----------------------------------------------------------------------- #
# Read all inputs
with open('vasp_settings.json') as json_file:
    vasp_settings = json.load(json_file)
    json_file.close()

kpd = vasp_settings.pop('kpd')

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

MLIP = MLIP_settings.pop('MLIP')
fmax_MLIP = MLIP_settings.pop('fmax')
MLIP_only = MLIP_settings.pop('MLIP_only')

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
if MLIP == 'uma':
    try: # pop task name
        task_name = MLIP_settings.pop('task_name')
    except:
        task_name = 'omat'
    predictor = pretrained_mlip.get_predict_unit(**MLIP_settings)
    calc = FAIRChemCalculator(predictor, task_name=task_name)
else: # default to MACE
    calc = mace_mp(**MLIP_settings)
struct.calc = calc

relaxer.run(fmax = fmax_MLIP)

    # clear the hessian for VASP
os.remove(restart_file)

if MLIP_only == False:
    # ------------------------------------------------------------- #
    # VASP RELAXATION
        # KPOINTS
    kpts_list = safe_kgrid_from_cell_volume(struct, kpd)

    calc = Vasp(**vasp_settings, kpts = kpts_list)
    struct.calc = calc

    relaxer.run(fmax = fmax)

    # ----------------------------------- #
    # High-Res DOS calculation:
    struct_dos = struct.copy()
    vasp_settings['ismear'] = -5
    vasp_settings['lelf'] = True
    struct_dos.calc = Vasp(**vasp_settings, kpts = kpts_list)
    struct_dos.get_potential_energy()

# ----------------------------------- #
# Write out relevant parameters
write('POSCAR_relaxed', struct, format = 'vasp')
struct.calc.write_json('calc_state.json')