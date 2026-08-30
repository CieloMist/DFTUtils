# ----------------------------------------------------------------------- #
# Strain sensitivity of a transformation barrier, from three converged geometries.
#
# Inputs, all read from the working directory:
#   Barrier.traj                       3 frames, ordered [initial, saddle, final]
#   mace_settings.json                 passed to mace_mp()
#   barrier_sensitivity_settings.json  see below
#
# barrier_sensitivity_settings.json keys (all optional except 'order'):
#   order              1 -> A_fwd / A_rxn only.  2 -> also the activation elastic tensor.
#   dispersion         include D3 in every derivative.            default true
#   atomic             per-atom virial decomposition.             default true
#   fmax_tol           free-atom stationarity gate, eV/A.         default 0.002
#   check_virial       finite-difference the displacement hook.   default true
#   check_eigenvalues  require (0, 1, 0) negative eigenvalues.    default true
#   directory_suffix   writes into Barrier_Sensitivity_<suffix>.  default none
#   results_file       output name.  default barrier_sensitivity_results.json
#
# The saddle in Barrier.traj should come from a dimer refinement, not from a NEB climbing
# image -- the envelope-theorem error scales directly with the residual force, and a band
# is only as converged as the band.  All three frames must share one cell and one FixAtoms
# set; _assert_consistent enforces both.

## EXAMPLE SETTINGS: 
# {"order": 2, "dispersion": true, "atomic": true, "fmax_tol": 0.002,
# "check_virial": true, "check_eigenvalues": true}
# ----------------------------------------------------------------------- #

# Imports
    # System
import os
import sys
import json

    # System +
import numpy as np
import torch

    # File io
from ase.io import read, write

    # MACE
from mace.calculators import mace_mp

    # Mine
from DFTUtils import make_directories_from_list
import BarrierSensitivity as bs

# ----------------------------------------------------------------------- #
# Read in settings
with open('barrier_sensitivity_settings.json') as json_file:
    settings = json.load(json_file)
    json_file.close()

order = int(settings.pop('order', 1))
assert order in (1, 2), f"order must be 1 or 2, got {order}"

use_d3 = settings.pop('dispersion', True)
atomic = settings.pop('atomic', True)
fmax_tol = settings.pop('fmax_tol', 2e-3)
check_virial = settings.pop('check_virial', True)
check_eigenvalues = settings.pop('check_eigenvalues', True)
directory_suffix = settings.pop('directory_suffix', None)

    # Read MACE Settings
with open('mace_settings.json') as json_file:
    mace_settings = json.load(json_file)
    json_file.close()

mace_settings.pop('fmax', None)              # not a mace_mp kwarg
head = mace_settings.get('head')

# ----------------------------------- #
# Make own directory
dirlist = ['Barrier_Sensitivity' if directory_suffix == None
           else 'Barrier_Sensitivity_' + directory_suffix]
make_directories_from_list(dirlist, delete = False)
os.chdir(dirlist[0])

# ---------------------------------- #
# Read the three stationary points
images = read('../Barrier.traj@:')
assert len(images) == 3, (
    f"Barrier.traj holds {len(images)} frames, expected exactly 3 (initial, saddle, final)"
)
atoms_i, atoms_s, atoms_f = images

# Frame 1 must be the highest. A file assembled in the wrong order sails through every
# other check and just reports a negative barrier with a sign-flipped A_fwd.
energies = np.array([img.get_potential_energy() for img in images])
assert energies[1] > energies[0] and energies[1] > energies[2], (
    f"frame 1 is not the highest energy (relative energies {energies - energies[0]}) "
    f"-- Barrier.traj must be ordered [initial, saddle, final]"
)
print(f"barrier fwd = {energies[1] - energies[0]:.4f} eV   "
      f"rev = {energies[1] - energies[2]:.4f} eV   "
      f"E_rxn = {energies[2] - energies[0]:.4f} eV   [as stored in Barrier.traj]")

# ----------------------------------- #
# Set MACE Calculator
calc = mace_mp(**mace_settings)

model = bs.load_mace(calc, device = "cpu")       # unwraps SumCalculator automatically
# MUST be the head the geometries were converged with. mace-mh-1 carries six heads and
# heads[0] is not the usual choice -- picking the wrong one silently evaluates a different
# functional, and every internal consistency check still passes.
batcher = bs.MaceBatcher(model, head = head)
print(f"head in use: {batcher.head}  (model heads: {batcher.heads})")

d3 = bs.D3Correction.from_calculator(calc) if use_d3 else None

# ---------------------------------- #
# Validate before trusting anything
if check_virial:
    bs.fd_check_virial(model, batcher, atoms_i)
    if d3 is not None:
        bs.fd_check_d3_virial(d3, atoms_i)

if check_eigenvalues:
    for lbl, a, expect in (("initial", atoms_i, 0), ("saddle", atoms_s, 1),
                           ("final", atoms_f, 0)):
        n_neg, _ = bs.n_negative_eigenvalues(model, batcher, a, d3 = d3)
        print(f"  {lbl:>7}: {n_neg} negative eigenvalue(s), expected {expect}")
        assert n_neg == expect, f"{lbl} is not a proper stationary point"

# ---------------------------------- #
# First order
results = bs.barrier_sensitivities(model, batcher, atoms_i, atoms_s, atoms_f,
                                   d3 = d3, atomic = atomic, fmax_tol = fmax_tol)

# ---------------------------------- #
# Second order
elastic = None
if order >= 2:
    elastic = bs.activation_elastic_tensor(model, batcher, atoms_i, atoms_s, d3 = d3)

# ---------------------------------- #
# Write results for the notebook
metadata = dict(
    head = batcher.head,
    includes_d3 = d3 is not None,
    mace_settings = mace_settings,
    cell = np.asarray(atoms_i.get_cell()).tolist(),
    n_atoms = len(atoms_i),
    n_free = int(bs._free_mask(atoms_i).sum()),
)
path = bs.write_results(results, destination = 'barrier_sensitivity_results.json', elastic = elastic,
                        extra = metadata)
print(f"\nwrote {os.path.join(os.getcwd(), path)}")
print("read it back with:  import BarrierSensitivity as bs; "
      "r = bs.read_results('<path>')")
