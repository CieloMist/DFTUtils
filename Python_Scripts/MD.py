# ------------------------------- #
# Imports
    # System
import os
import json
import shutil as sh

    # File io
from ase.io import read, write

# ASE
from ase.calculators.vasp import Vasp
from asekpd import safe_kgrid_from_cell_volume
from mace.calculators import mace_mp

# Mine
from DFTUtils import write_vasp_settings, write_settings_json, get_strain, render_chg_slice
from DFTUtils import copy_files_from_DFTUtilities, remove_files, make_directories_from_list

# ----------------------------------- #
# Read in VASP settings
os.environ["VASP_PP_PATH"] = "/projects/b1027/Pseudopotentials.64"
os.environ["VASP_COMMAND"] = "mpirun -n $SLURM_NTASKS vasp_std"

# Read AIMD Settings:
with open('MD_settings.json') as json_file:
    MD_settings = json.load(json_file)
    json_file.close()

temperature_K = MD_settings.pop('temperature_K')
n_steps = MD_settings.pop('n_steps')
AIMD = MD_settings.pop('AIMD', True)

if AIMD:
    # Read Input Settings - all INCAR and POTCAR settings specified here
    with open('vasp_settings.json') as json_file:
        vasp_settings = json.load(json_file)
        json_file.close()

    kpd = vasp_settings.pop('kpd')

else:
    with open('mace_settings.json') as json_file:
        mace_settings = json.load(json_file)
        json_file.close()

# --------------------------------------------------------------------------- #
# Set Structure
struct= read(r'Initial.traj')

# ----------------------------------------------------------------#
# Set Calculator
if AIMD:
    # KPOINTS
    kpts_list = kpd
    if isinstance(kpd, int):
        kpts_list = safe_kgrid_from_cell_volume(struct, kpd)
    calc = Vasp(**vasp_settings, kpts = kpts_list)
else:
    calc = mace_mp(**mace_settings)

struct.calc = calc
# ----------------------------------------------------------------#
# MD
from ase.md import Langevin, MDLogger
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# setup
MaxwellBoltzmannDistribution(struct, temperature_K = temperature_K)
integrator = Langevin(struct, temperature_K = temperature_K, **MD_settings)

# attach logger
log = MDLogger(integrator, struct, 'MD.log', header=False, stress=True, peratom=True, mode="w")
integrator.attach(log, interval=1)

# write trajectory
traj = Trajectory('MD' + '.traj', 'w', struct)
integrator.attach(traj, interval=1)

    # Run Simulation
integrator.run(n_steps)
