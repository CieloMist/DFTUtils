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
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.optimize import BFGS, BFGSLineSearch, GPMin, MDMin, GoodOldQuasiNewton
    # Filters/Masks
from ase.filters import FrechetCellFilter
from ase.filters import StrainFilter
# from ase.stressbox import stressbox
    # File io
from ase.io import read, write
from kgrid import safe_kgrid_from_cell_volume

# ----------------------------------- #
# Read in settings
    # Read Input Settings - all INCAR and POTCAR settings specified here
with open('qe_settings.json') as json_file:
    qe_settings = json.load(json_file)
    json_file.close()

kpd = qe_settings.pop('kpd')
pseudopotentials = qe_settings.pop('pseudopotentials')
pseudo_dir = qe_settings.pop('pseudo_dir')
additional_cards = qe_settings.pop('additional_cards', None)

# ----------------------------------- #
# Read in filter settings
with open('filter_settings.json') as json_file:
    filter_settings = json.load(json_file)
    json_file.close()

filter_type = filter_settings.pop('filter', 'Frechet')
optimizer = filter_settings.pop('optimizer', 'BFGS')
fmax = filter_settings.pop('fmax', 0.005)
restart = filter_settings.pop('restart', False)

# ----------------------------------------------------------------------- #
# Calculation Details
    # Set Initial Structure
struct = read('Initial.traj', format = 'traj')

if (restart == True) and (os.path.exists(os.getcwd() + '/relax.traj')):
    struct = read('relax.traj', format = 'traj')

    # KPOINTS
kpts_list = kpd
if isinstance(kpd, int):
    kpts_list = safe_kgrid_from_cell_volume(struct, kpd)

profile_settings = {'command': 'mpirun pw.x -pd .true.', 'pseudo_dir': pseudo_dir} # change to explicit number of cores here
profile = EspressoProfile(**profile_settings)

# ----------------------------------- #
# Pre-calculation - calculates wavefunctions once to allow restart from file later.
if restart == False:
    calc = Espresso(profile = profile,
                    pseudopotentials = pseudopotentials,
                    input_data = qe_settings,
                    additional_cards=additional_cards,
                    kpts = kpts_list)

    struct.calc = calc

    # get initial potential energy to allow for restart from wavefunctions during relaxation
    struct.get_potential_energy()

# ----- #
# second run - modify QE settings to allow restart from file
qe_settings['electrons']['startingwfc'] = 'file'
qe_settings['electrons']['startingpot'] = 'file'

calc = Espresso(profile = profile,
                pseudopotentials = pseudopotentials,
                input_data = qe_settings,
                additional_cards=additional_cards,
                kpts = kpts_list)

struct.calc = calc

# ----------------------------------- #
# Set cell filter to optimize cell positions too
if filter_type == 'Frechet':
    struct_opt = FrechetCellFilter(struct, **filter_settings)
elif filter_type == 'strain':
    struct_opt = StrainFilter(struct, **filter_settings)
else:
    struct_opt = struct.copy() # set no filter if no filter is provided

# ---------------------------------- #
# Set Optimizer
if optimizer == 'BFGSLineSearch':
    relaxer = BFGSLineSearch(struct_opt, trajectory = 'relax.traj', restart='BFGSLS_hessian.json')
elif optimizer == 'GPMin':
    relaxer = GPMin(struct_opt, trajectory = 'relax.traj', restart='gaussian_process.json')
elif optimizer == 'GOQN':
    relaxer = GoodOldQuasiNewton(struct_opt, trajectory = 'relax.traj', restart='GOQN.json')
elif optimizer == 'MDMin':
    relaxer = MDMin(struct_opt, trajectory = 'relax.traj', restart = 'MDMin_restart.json')
else:
    relaxer = BFGS(struct_opt, trajectory = 'relax.traj', restart='BFGS_hessian.json')
    
    # Run Relaxation
relaxer.run(fmax = fmax)

# ----------------------------------- #
# Do DOS Projection
# rerun calculation with tetrahedron method
qe_settings['system']['occupations'] = 'tetrahedra'

calc = Espresso(profile = profile,
                pseudopotentials = pseudopotentials,
                input_data = qe_settings,
                additional_cards=additional_cards,
                kpts = kpts_list)

struct.calc = calc
struct.get_potential_energy()

projwfc_settings = {'projwfc': {'prefix': qe_settings['prefix'],
                                'outdir': qe_settings['outdir'],
                                'DeltaE': 0.005}}
from ase.io.espresso import write_fortran_namelist
with open('projwfc.in', 'w') as file:
    write_fortran_namelist(file, input_data=projwfc_settings)

subprocess.run('mpirun -np 1 projwfc.x -pd .true. -inp projwfc.in > projwfc.out', shell=True)