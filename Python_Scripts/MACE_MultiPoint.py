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
with open('MLIP_settings.json') as json_file:
    MLIP_settings = json.load(json_file)
    json_file.close()

# ----------------------------------------------------------------------- #
# Calculation Details
    # Set Initial Structure
images = read('images.traj@:', format = 'traj')

# ----------------------------------- #
# Set Calculator

for image in images:
    calc = mace_mp(**MLIP_settings)
    image.calc = calc
    image.get_forces()

write('Images_Final.traj', images, format = 'traj')

