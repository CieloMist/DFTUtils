# ----------------------------------------------------------------------- #
# Imports
    # System
import os
import sys
import json
import subprocess

# ASE
from ase.calculators.espresso import Espresso, EspressoProfile

    # File io
from ase.io import read, write
from kgrid import safe_kgrid_from_cell_volume # TEMPORARILY CHANGES FROM ASECLT TO AMLT

# ----------------------------------- #
    # Read Input Settings - all INCAR and POTCAR settings specified here
with open('qe_settings.json') as json_file:
    qe_settings = json.load(json_file)
    json_file.close()

kpd = qe_settings.pop('kpd')
pseudopotentials = qe_settings.pop('pseudopotentials')
pseudo_dir = qe_settings.pop('pseudo_dir')
additional_cards = qe_settings.pop('additional_cards', None)

# ----------------------------------------------------------------------- #
# Calculation Details
    # Set Initial Structure
struct = read('Initial.traj', format = 'traj')

    # KPOINTS
kpts_list = safe_kgrid_from_cell_volume(struct, kpd)

profile_settings = {'command': 'mpirun pw.x -pd .true.', 'pseudo_dir': pseudo_dir} # change for efficiency later if you need
profile = EspressoProfile(**profile_settings)

# ----------------------------------- #
# High-Res DOS calculation:
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

write('Final.traj', struct, format = 'traj')

projwfc_settings = {'projwfc': {'prefix': qe_settings['prefix'],
                                'outdir': qe_settings['outdir'],
                                'DeltaE': 0.01}}

from ase.io.espresso import write_fortran_namelist
with open('projwfc.in', 'w') as file:
    write_fortran_namelist(file, input_data=projwfc_settings)

subprocess.run('mpirun -np 1 projwfc.x -pd .true. -inp projwfc.in > projwfc.out', shell=True)

