# ----------------------------------------------------------------------- #
# Imports
    # System
import os
import sys
import json

# ASE
from mace.calculators import mace_mp

    # File io
from ase.io import read, write
from asekpd import safe_kgrid_from_cell_volume

# ----------------------------------- #
# Read in MACE
    # Read Input Settings - all INCAR and POTCAR settings specified here
with open('mace_settings.json') as json_file:
    mace_settings = json.load(json_file)
    json_file.close()

# ----------------------------------------------------------------------- #
# Calculation Details
    # Set Initial Structure
images = read('Images.traj@:', format = 'traj')

# ----------------------------------- #
# Set Calculator

for image in images:
    calc = mace_mp(**mace_settings)
    image.calc = calc
    image.get_potential_energy()

write('Images_Final.traj', images, format = 'traj')

