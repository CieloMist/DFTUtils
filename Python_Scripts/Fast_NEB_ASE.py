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
from ase.optimize import BFGS, MDMin, FIRE2, LBFGS
from ase.io.trajectory import Trajectory

    # NEB
from ase.mep import NEB
from ase.mep.dyneb import DyNEB

    # File io
from ase.io import read, write

#MACE
from mace.calculators import mace_mp

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
read_images = neb_settings.pop('read_images', False)

# ---------------------- #
    # Read MACE Settings
with open('mace_settings.json') as json_file:
    mace_settings = json.load(json_file)
    json_file.close()

fmax_mace = mace_settings.pop('fmax', 0.05)

# ---------------------------------- #
# Set Initial Structure
initial = read('Initial.traj', format = 'traj')
final = read('Final.traj', format = 'traj')

# initial.calc = mace_mp(**mace_settings)
# final.calc = mace_mp(**mace_settings)

# initial.get_potential_energy()
# final.get_potential_energy()

# ---------------------------------- #
# Initialize Images and NEB
if restart == True:
    images = read('NEB.traj@-' + str(n_images) + ':')
    neb = DyNEB(images, **neb_settings)

elif read_images == True:
    images = read('images.traj@:')
    neb = DyNEB(images, **neb_settings)

else:
    images = [initial]
    images += [initial.copy() for i in range(n_images-2)]
    images += [final]

    neb = DyNEB(images, **neb_settings)
    neb.interpolate(method='idpp')

# ----------------------------------- #
# Set MACE Calculator
for image in neb.images:
    # Set Calculator
    calc = mace_mp(**mace_settings)
    image.calc = calc

    # image.get_potential_energy()

neb.images[0].get_forces()
neb.images[-1].get_forces()

# ---------------------------------- #
# Set relaxer
if optimizer == 'BFGS':
    relaxer = BFGS(neb, trajectory = 'NEB.traj')
elif optimizer == 'LBFGS':
    relaxer = LBFGS(neb, trajectory = 'NEB.traj')
elif optimizer == 'MDMin':
    relaxer = MDMin(neb, trajectory = 'NEB.traj')
elif optimizer == 'FIRE' or optimizer == 'FIRE2':
    relaxer = FIRE2(neb, trajectory = 'NEB.traj', use_abc = True, dt = 0.05, maxstep = 0.05, dtmax = 0.1)

# # Initial and final states
neb.images[0].get_potential_energy()
neb.images[-1].get_potential_energy()

relaxer.run(fmax_mace) # fmax = fmax_mace
