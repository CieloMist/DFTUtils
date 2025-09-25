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
from ase.optimize import BFGS, BFGSLineSearch, GPMin
    # Filters/Masks
from ase.filters import FrechetCellFilter
from ase.filters import StrainFilter
    # File io
from ase.io import read, write
# Pymatgen Stuff
from pymatgen.io.vasp import Kpoints
from pymatgen.io.lobster.inputs import Lobsterin

# ----------------------------------- #
# VASP Path Definitions
os.environ["VASP_PP_PATH"] = "/projects/b1027/Pseudopotentials.64"
os.environ["VASP_COMMAND"] = "mpirun -n $SLURM_NTASKS vasp_std"

# ----------------------------------------------------------------------- #
# Calculation Details

# Read json of lobster settings
with open('lobster_settings.json') as json_file:
    lobster_settings = json.load(json_file)
    json_file.close()

# ----------------------------------- #
# Ready lobster calculation files
lobsterin = Lobsterin.standard_calculations_from_vasp_files(POSCAR_input = '../POSCAR',
                                                            INCAR_input = '../INCAR',
                                                            POTCAR_input = '../POTCAR')

lobsterin.write_POSCAR_with_standard_primitive(POSCAR_input = '../POSCAR', POSCAR_output= 'POSCAR')
lobsterin.write_INCAR(incar_input='../INCAR', incar_output='INCAR', poscar_input = 'POSCAR')
lobsterin.write_KPOINTS(POSCAR_input = 'POSCAR', reciprocal_density = 100, KPOINTS_output = 'KPOINTS') # Current folder

# Adjust Settings
for key in lobster_settings.keys():
    lobsterin[key] = lobster_settings[key]
    # Write
lobsterin.write_lobsterin()
