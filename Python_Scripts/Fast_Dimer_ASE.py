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

fmax = dimer_settings.pop('fmax', 0.005)
directory_suffix = dimer_settings.pop('directory_suffix', None)
max_steps = dimer_settings.pop('steps', 1000)

# Seeded externally (DFTUtils.get_neb_eigenmode) when refining a NEB climbing image.
# Absent -> fall back to the displacement route below.
eigenmodes = dimer_settings.pop('eigenmodes', None)

if eigenmodes != None:
    eigenmodes = [np.asarray(mode, dtype = float) for mode in eigenmodes]

    # Read MACE Settings
with open('mace_settings.json') as json_file:
    mace_settings = json.load(json_file)
    json_file.close()

# ----------------------------------- #
# Make sub-directory
dirlist = ['Dimer' if directory_suffix == None else 'Dimer_' + directory_suffix]
make_directories_from_list(dirlist, delete = True)
os.chdir(dirlist[0])

# ---------------------------------- #
# Set Initial Structure
displacement_vector = None

if eigenmodes != None:
    # Seeded from a NEB climbing image: the geometry is already AT the saddle, so there is
    # nothing to displace. We only need the mode, which came in through the settings.
    struct = read('../Initial_Dimer.traj', format = 'traj')
else:
    struct = read('../displacements.traj@-1', format = 'traj')
    displacements = read('../displacements.traj@:')
    displacement_vector = (displacements[1].get_positions()
                           - displacements[0].get_positions())

# Calculate using MACE
calc = mace_mp(**mace_settings)
struct.calc = calc
struct.get_potential_energy()

print(f'starting fmax = {np.linalg.norm(struct.get_forces(), axis = 1).max():.6f} eV/A')

# Set up the dimer
with DimerControl(**dimer_settings) as d_control:
    # Passing eigenmodes here makes DimerControl's initial_eigenmode_method irrelevant --
    # ASE ignores it when a mode is supplied at construction.
    d_atoms = MinModeAtoms(struct, d_control, eigenmodes = eigenmodes)

    # Displace ONLY when starting from a minimum.
    if displacement_vector is not None:
        d_atoms.displace(displacement_vector = displacement_vector)

    # Converge to a saddle point
    with MinModeTranslate(
        d_atoms, trajectory='dimer_evolution.traj', logfile='translation.log'
    ) as dim_rlx:
        dim_rlx.run(fmax=fmax, steps=max_steps)

# ---------------------------------- #
# Report
    # d_atoms.get_forces() returns the DIMER-modified force (component along the mode
    # inverted). The reflection preserves the global 3N norm but redistributes it between
    # atoms, so the per-atom max differs from the true one. Report both.
print(f'final fmax (dimer) = {np.linalg.norm(d_atoms.get_forces(), axis = 1).max():.6f} eV/A')
print(f'final fmax (true)  = {np.linalg.norm(struct.get_forces(), axis = 1).max():.6f} eV/A')
print(f'final curvature    = {d_atoms.get_curvature():.6f} eV/A^2')
if d_atoms.get_curvature() > 0:
    print('  WARNING: curvature is POSITIVE -- not an index-1 saddle. The seed mode '
          'probably missed; check translation.log.')

# Snapshot into a SinglePointCalculator before writing. Writing d_atoms.atoms directly
# stores neither energy nor forces: the dimer leaves pending position changes, so the
# trajectory writer cannot pull the properties off the live calculator.
from ase.calculators.singlepoint import SinglePointCalculator

saddle = struct.copy()
saddle.calc = SinglePointCalculator(saddle,
                                    energy = struct.get_potential_energy(),
                                    forces = struct.get_forces())
write('saddle.traj', saddle)
