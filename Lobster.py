# ----------------------------------------------------------------------- #
# Imports
    # System
import os
import sys
import json
import subprocess
import shutil as sh
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
from ase.build import niggli_reduce
from asekpd import safe_kgrid_from_cell_volume
# Pymatgen Stuff
from pymatgen.io.lobster.inputs import Lobsterin
from pymatgen.core import Structure
# Mine
from DFTUtils import make_directories_from_list

# ----------------------------------- #
# VASP Path Definitions
os.environ["VASP_PP_PATH"] = "/projects/b1027/Pseudopotentials.64"
os.environ["VASP_COMMAND"] = "mpirun -n $SLURM_NTASKS vasp_std"

# ----------------------------------------------------------------------- #
# Read json of lobster settings
with open('lobster_settings.json') as json_file:
    lobster_settings = json.load(json_file)
    json_file.close()

directory_suffix = lobster_settings.pop('directory_suffix')

# Read VASP settings
if not os.path.isfile('vasp_settings.json'):
    sh.copy('../vasp_settings.json', '.')

with open('vasp_settings.json') as json_file:
    vasp_settings = json.load(json_file)
    json_file.close()

kpd = vasp_settings.pop('kpd')

# ----------------------------------- #
# Ready lobster calculation files
dirlist = ['Lobster' if directory_suffix == None else 'Lobster_' + directory_suffix]
make_directories_from_list(dirlist, delete = False)
os.chdir(dirlist[0])

# ----------------------------------- #
# Ready lobster calculation files
lobsterin = Lobsterin.standard_calculations_from_vasp_files(POSCAR_input = '../POSCAR',
                                                            INCAR_input = '../INCAR',
                                                            POTCAR_input = '../POTCAR')

# symmetrize structure
struct = read('../relax.traj')
niggli_reduce(struct)
write('POSCAR', struct)

lobsterin.write_INCAR(incar_input='../INCAR', incar_output='INCAR', poscar_input = 'POSCAR')

# Adjust Settings
for key in lobster_settings.keys():
    lobsterin[key] = lobster_settings[key]
    # Write
lobsterin.write_lobsterin()

# ----------------------------------- #
# Run DFT Calculation
struct = read('POSCAR', format = 'vasp')

    # KPOINTS
kpts_list = safe_kgrid_from_cell_volume(struct, kpd)

    # Read in INCAR settings - change the necessary settings for lobster
nbands = lobsterin._get_nbands(structure=Structure.from_ase_atoms(struct))

vasp_settings['nbands'] = nbands
vasp_settings['isym'] = 0
vasp_settings['lwav'] = True
vasp_settings['nsw'] = 0
vasp_settings['ismear'] = -5
vasp_settings['lelf'] = True

    # Set Calc
calc = Vasp(**vasp_settings, kpts = kpts_list)
struct.calc = calc

    # Run calculation:
struct.get_potential_energy()

# ----------------------------------- #
# Run LOBSTER
subprocess.run('/projects/p32212/Software_LifeEasy/lobster-5.1.1/lobster-5.1.1')