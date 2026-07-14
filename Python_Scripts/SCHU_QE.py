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
from ase.stressbox import stressbox
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

with open('SCHU_settings.json') as json_file:
    SCHU_settings = json.load(json_file)
    json_file.close()
# ----------------------------------------------------------------------- #
# Calculation Details
    # Set Initial Structure
struct = read('Initial.traj', format = 'traj')

    # KPOINTS
kpts_list = safe_kgrid_from_cell_volume(struct, kpd)

profile_settings = {'command': 'mpirun pw.x -pd .true.', 'pseudo_dir': pseudo_dir} # change for efficiency later if you need
profile = EspressoProfile(**profile_settings)

# ----------------------------------- #
# Set Calculator
calc = Espresso(profile = profile,
                pseudopotentials = pseudopotentials,
                input_data = qe_settings,
                additional_cards=additional_cards,
                kpts = kpts_list)

struct.calc = calc
struct.get_potential_energy()

# ----------------------------------- #
# hp.x
profile_settings = {'command': 'hp.x', 'pseudo_dir': pseudo_dir} # change for efficiency later if you need
profile = EspressoProfile(**profile_settings)

# ----------------------------------- #
# Set Calculator

# SCHU_settings['inputhp']['nq1'] = kpts_list[0]
# SCHU_settings['inputhp']['nq2'] = kpts_list[1]
# SCHU_settings['inputhp']['nq3'] = kpts_list[2]

SCHU_settings['inputhp']['nq1'] = 2
SCHU_settings['inputhp']['nq2'] = 2
SCHU_settings['inputhp']['nq3'] = 2

from ase.io.espresso import write_fortran_namelist
with open('hp.in', 'w') as file:
    write_fortran_namelist(file, input_data=SCHU_settings)

subprocess.run('mpirun hp.x -pd .true. -inp hp.in > hp_params.out', shell=True)
