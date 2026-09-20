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

# mine
from select_diverse_frames import mace_descriptors, save_descriptors, load_descriptors, fps_frames, distance_to_reference, strip_calc

# ----------------------------------- #
# Read in MACE
    # Read Input Settings - all INCAR and POTCAR settings specified here
with open('mace_settings.json') as json_file:
    mace_settings = json.load(json_file)
    json_file.close()

with open('select_training_frames_settings.json') as json_file:
    select_training_frames_settings = json.load(json_file)
    json_file.close()

# add a mode here that allows for calculation of reference descriptors output to a file path. then, instead of recalculating them every time, just read them.

replay_structure_path = select_training_frames_settings.pop('replay_structure_path', None)
replay_descriptor_path = select_training_frames_settings.pop('replay_descriptor_path', None)

reference_data = select_training_frames_settings.pop('reference_data', False)

# ----------------------------------- #
# load reference data and replay data
if reference_data:
    reference_images = read('reference_images.traj@:')
    
    calc = mace_mp(**mace_settings)
    reference_descriptors = mace_descriptors(reference_images, calc)
else:
    reference_images = None
    reference_descriptors = None

if replay_structure_path != None:
    replay_descriptors = load_descriptors(replay_descriptor_path)
    replay_images = read(replay_structure_path, ':')

# TEMPORARY BLOCK::
calc = mace_mp(**mace_settings)

# once: Sn-bearing and H-bearing parts of the replay subset -> descriptors (~0.5 h on an 8-core short node)
# replay = read("/projects/p32212/Collaborator_Projects/Duncan_Sn/MLIP_Fine_Tuning/Replay/replay_mh1_SnH_plus_10k.xyz", ":")
# save_descriptors("/projects/p32212/Collaborator_Projects/Duncan_Sn/MLIP_Fine_Tuning/Replay/replay_mh1_SnH_plus_10k.npz", mace_descriptors(replay, calc))
# # END TEMP BLOCK

# ----------------------------------------------------------------------- #
# Calculation Details
traj = read('training_images.traj@:')          # drop equilibration, stride
calc = mace_mp(**mace_settings)
desc = mace_descriptors(traj, calc)      # list of (n_atoms, 1024) invariant arrays
idx, scores = fps_frames(desc, traj, reference=reference_descriptors, reference_frames = reference_images, **select_training_frames_settings)

novelty_all = distance_to_reference(desc, traj, replay_descriptors, replay_images)
novelty = novelty_all[idx]

# ----------------------------------- #
# write
for ind, image in enumerate(traj):
    image.info['novelty'] = novelty_all[ind]
    if ind in idx:
        image.info['selected'] = True
    else:
        image.info['selected'] = False
write('selected_images.traj', [strip_calc(traj[i]) for i in idx])   # drops MACE E/F/stress
