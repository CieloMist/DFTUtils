# ----------------------------------------------------------------------- #
# Imports
    # System
import os
import sys
import json
import subprocess

    # System +
import numpy as np

    # Dimer
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

    # File io
from ase.io import read, write

    #MACE
from mace.calculators import mace_mp

    # Mine
from DFTUtils import make_directories_from_list

# ----------------------------------------------------------------------- #
    # Read Dimer Settings
with open('dimer_settings.json') as json_file:
    dimer_settings = json.load(json_file)
    json_file.close()
fmax = dimer_settings.pop('fmax')

directory_suffix = dimer_settings.pop('directory_suffix', None)
fmax = dimer_settings.pop('fmax', 0.005)

    # Read MACE Settings
with open('MLIP_settings.json') as json_file:
    MLIP_settings = json.load(json_file)
    json_file.close()

# ----------------------------------- #
# Make sub-directory
dirlist = ['Dimer' if directory_suffix == None else 'Dimer_' + directory_suffix]
make_directories_from_list(dirlist, delete = True)
os.chdir(dirlist[0])

# ---------------------------------- #
# Set Initial Structure
struct = read('../displacements.traj@-1', format = 'traj')

# Calculate using MACE
calc = mace_mp(**MLIP_settings)
struct.calc = calc
struct.get_potential_energy()

# Set up the dimer
with DimerControl(**dimer_settings) as d_control:
    d_atoms = MinModeAtoms(struct, d_control)

    # Displace the atoms
    displacements = read('../displacements.traj@:')
    displacement_vector = displacements[1].get_positions() - displacements[0].get_positions()
    d_atoms.displace(displacement_vector = displacement_vector)

    # d_atoms.rattle() # random for now

    # Converge to a saddle point
    with MinModeTranslate(
        d_atoms, trajectory='dimer_method.traj', logfile='translation.log'
    ) as dim_rlx:
        dim_rlx.run(fmax=fmax)
