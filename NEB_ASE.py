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
from ase.optimize import BFGS, BFGSLineSearch, GPMin, LBFGS, MDMin, FIRE
    # NEB
from ase.mep import NEB
    # Filters/Masks
from ase.filters import FrechetCellFilter
from ase.filters import StrainFilter
    # File io
from ase.io import read, write
from asekpd import safe_kgrid_from_cell_volume
# Pymatgen Stuff
from pymatgen.io.vasp import Kpoints

# ----------------------------------- #
# VASP Path Definitions
os.environ["VASP_PP_PATH"] = "/projects/b1027/Pseudopotentials.64"
os.environ["VASP_COMMAND"] = "mpirun -n $SLURM_NTASKS vasp_std"

# ----------------------------------------------------------------------- #
# Read in settings
    # Read NEB Settings
with open('neb_settings.json') as json_file:
    neb_settings = json.load(json_file)
    json_file.close()
restart = neb_settings.pop('restart')
n_images = neb_settings.pop('n_images')
fmax = neb_settings.pop('fmax')
optimizer = neb_settings.pop('optimizer')

    # Read DFT Settings
with open('vasp_settings.json') as json_file:
    vasp_settings = json.load(json_file)
    json_file.close()
kpd = vasp_settings.pop('kpd')

# ---------------------------------- #
# Set Initial Structure
initial = read('Initial.traj', format = 'traj')
final = read('Final.traj', format = 'traj')

kpts_list = safe_kgrid_from_cell_volume(initial, kpd) # play around with maximizing density between structures later

# ---------------------------------- #
# Initialize Images and NEB
if restart == True:
    images = read('NEB.traj@-' + str(n_images) + ':')   
    neb = NEB(images, **neb_settings)
else:
    images = [initial]
    images += [initial.copy() for i in range(n_images-2)]
    images += [final]
    
    neb = NEB(images, **neb_settings)
    neb.interpolate()
# ----------------------------------- #
# Set Calculator for images
for image in images:
    # Set Calculator
    calc = Vasp(**vasp_settings, kpts = kpts_list)
    image.calc = calc

# ---------------------------------- #
# Calculate initial and final energies
initial.get_potential_energy()
final.get_potential_energy()

# ---------------------------------- #
# Set Optimizer
if optimizer == 'GPMin':
    relaxer = GPMin(neb, trajectory = 'NEB.traj')
elif optimizer == 'BFGS':
    relaxer = BFGS(neb, trajectory = 'NEB.traj')
elif optimizer == 'LBFGS':
    relaxer = LBFGS(neb, trajectory = 'NEB.traj')
elif optimizer == 'MDMin':
    relaxer = MDMin(neb, trajectory = 'NEB.traj')
else:
    relaxer = FIRE(neb, trajectory = 'NEB.traj')

    # Run Optimization
relaxer.run(fmax = fmax)
# ---------------------------------- #
# Write out calculator states
for ind, image in enumerate(images):
    image.calc.write_json('calc_state_' + str(ind) + '.json')
