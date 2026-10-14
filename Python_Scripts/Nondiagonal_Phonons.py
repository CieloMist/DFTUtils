# ----------------------------------------------------------------------- #
# Nondiagonal supercell phonons (Lloyd-Williams & Monserrat, PRB 92, 184301)
# using Quesadilla's python API, with VASP forces through ASE.
#
# Inputs (in the submission directory):
#   Initial_Phonons.traj      structure (any cell; reduced to the standard primitive)
#   vasp_settings.json        ASE Vasp kwargs + 'kpd'
#   quesadilla_settings.json  {"grid": [4, 4, 4],                 # q-grid (required)
#                              "directory_suffix": null,
#                              "structure_file": "Initial_Phonons.traj",
#                              "displacement_distance": 0.01,
#                              "is_plusminus": "auto",
#                              "symprec": 1e-5,
#                              "subtract_residual_forces": false,
#                              "dry_run": false}                 # set up dirs only, no VASP
#
# Restartable: finished displacements (disp-XXX/Final.traj) are reused, so an
# out-of-walltime job can simply be resubmitted.
# ----------------------------------------------------------------------- #
# Imports
    # System
import os
import sys
import json
    # System +
import numpy as np
# ASE Stuff
from ase.calculators.vasp import Vasp
    # File io
from ase.io import read, write
from asekpd import safe_kgrid_from_cell_volume
# Phonopy stuff
from phonopy import Phonopy
# Quesadilla stuff
from quesadilla.cli import get_phonopy_prim
from quesadilla.supercells import SupercellGenerator
from quesadilla.dynmat import NondiagonalPhononCalculator
# Mine
from DFTUtils import make_directories_from_list, interchange_atoms_ase_phonopy

# ----------------------------------- #
# VASP Path Definitions
os.environ["VASP_PP_PATH"] = "/projects/b1027/Pseudopotentials.64"
os.environ["VASP_COMMAND"] = "mpirun -n $SLURM_NTASKS vasp_std"

# ----------------------------------------------------------------------- #
# Read json of quesadilla settings
with open('quesadilla_settings.json') as json_file:
    quesadilla_settings = json.load(json_file)
    json_file.close()

grid = np.array(quesadilla_settings.pop('grid'), dtype = int)
directory_suffix = quesadilla_settings.pop('directory_suffix', None)
structure_file = quesadilla_settings.pop('structure_file', 'Initial_Phonons.traj')
displacement_distance = quesadilla_settings.pop('displacement_distance', 0.01)
is_plusminus = quesadilla_settings.pop('is_plusminus', 'auto')
symprec = quesadilla_settings.pop('symprec', 1e-5)
subtract_residual_forces = quesadilla_settings.pop('subtract_residual_forces', False)
dry_run = quesadilla_settings.pop('dry_run', False)

# Read VASP settings
with open('vasp_settings.json') as json_file:
    vasp_settings = json.load(json_file)
    json_file.close()

kpd = vasp_settings.pop('kpd')

if vasp_settings.get('ispin', 1) == 2:
    print('WARNING: ispin = 2, but Quesadilla ignores magnetic moments when finding '
          'symmetry. Only safe if the magnetic order does not lower the symmetry.')

# ----------------------------------- #
# Ready quesadilla calculation files
dirlist = ['Nondiagonal_Phonons' if directory_suffix == None else 'Nondiagonal_Phonons_' + directory_suffix]
make_directories_from_list(dirlist, delete = False)
os.chdir(dirlist[0])

# ----------------------------------------------------------------------- #
# Functions
def get_forces(phonopy_atoms, directory, kpts_list):
    """
    Single point VASP forces on one (displaced) supercell in directory.
    Reuses directory/Final.traj from a previous run if it holds the same structure.
    """
    struct = interchange_atoms_ase_phonopy(phonopy_atoms)
    struct.set_pbc([1,1,1])

    final_file = os.path.join(directory, 'Final.traj')
    if os.path.isfile(final_file):
        final = read(final_file, format = 'traj')
        dfrac = final.get_scaled_positions(wrap = False) - struct.get_scaled_positions(wrap = False)
        dfrac -= np.rint(dfrac)
        if (len(final) != len(struct)) or (not np.allclose(dfrac, 0, atol = 1e-6)):
            raise RuntimeError(f'{final_file} does not match the displaced supercell Quesadilla '
                               'generated. Delete this calculation directory (stale from other settings?).')
        print(f'{directory}: reusing forces from Final.traj')
        return final.get_forces()

    write(os.path.join(directory, 'Initial.traj'), struct, format = 'traj')

    calc = Vasp(**vasp_settings, kpts = kpts_list, directory = directory)
    struct.calc = calc
    struct.get_potential_energy()

    # ASE does not raise on an unconverged SCF; don't let those forces into the FCs
    if not calc.converged:
        raise RuntimeError(f'VASP electronic SCF did not converge in {directory}.')

    write(final_file, struct, format = 'traj')
    return struct.get_forces()

# ----------------------------------------------------------------------- #
# Standard primitive cell: Quesadilla's symmetry/IBZ routines assume the spglib
# standard primitive, so the q-grid is defined w.r.t. PPOSCAR's reciprocal lattice.
initial_struct = read('../' + structure_file)
write('POSCAR', initial_struct, format = 'vasp', direct = True)
primitive, cell_info = get_phonopy_prim('POSCAR', calculator = 'vasp', symprec = symprec) # writes PPOSCAR
unitcell = cell_info['unitcell']

# ------------------------------------------------ #
# Nondiagonal supercells: minimum set covering the irreducible q-points of grid.
# Reuse quesadilla.toml on restart so the set cover can't change between jobs.
if not os.path.isfile('quesadilla.toml'):
    sc_gen = SupercellGenerator(primitive, grid)
    sc_gen.generate_supercells(minimize_supercells = True)
    sc_gen.to_toml('quesadilla.toml')
sc_gen = SupercellGenerator.from_toml('quesadilla.toml')
sc_gen.q_ibz = sc_gen.get_ibz() # Quesadilla bug: from_toml validates the IBZ against get_ibz() but never stores it

if not np.array_equal(sc_gen.grid, grid):
    raise ValueError(f'quesadilla.toml was made for grid {sc_gen.grid}, settings ask for {grid}. '
                     'Use a new directory_suffix or delete the old directory.')

# ------------------------------------------------ #
# Generate displacements for every supercell
sc_directories = [f'sc-{ind+1:03d}' for ind in range(len(sc_gen.sc_matrices))]
make_directories_from_list(sc_directories, delete = False)

nd_phonons = []
for sc_ind, sc_matrix in enumerate(sc_gen.sc_matrices):
    # phonopy's supercell matrix is the transpose of Quesadilla's (PRB 92, 184301 convention)
    phonons = Phonopy(unitcell, supercell_matrix = sc_matrix.T, symprec = symprec)
    phonons.generate_displacements(distance = displacement_distance, is_plusminus = is_plusminus)
    phonons.save(os.path.join(sc_directories[sc_ind], 'phonopy_disp.yaml'))
    write(os.path.join(sc_directories[sc_ind], 'SPOSCAR'), interchange_atoms_ase_phonopy(phonons.supercell), format = 'vasp', direct = True)
    nd_phonons.append(phonons)

n_calcs = sum(len(p.supercells_with_displacements) + int(subtract_residual_forces) for p in nd_phonons)
print('------------------------------------')
print(f'q-grid {grid.tolist()}: {len(sc_gen.q_ibz)} irreducible q-points, '
      f'{len(nd_phonons)} nondiagonal supercells, {n_calcs} VASP calculations total')
for sc_ind, phonons in enumerate(nd_phonons):
    print(f'{sc_directories[sc_ind]}: {len(phonons.supercell)} atoms, '
          f'{len(phonons.supercells_with_displacements)} displacements, '
          f'q = {np.round(sc_gen.q_comm[sc_ind], 4).tolist()}')
print('------------------------------------')

if dry_run:
    sys.exit(0)

# ----------------------------------------------------------------------- #
# Run DFT calculations, build force constants for each supercell
for sc_ind, phonons in enumerate(nd_phonons):
    supercells = phonons.supercells_with_displacements

    # KPOINTS - same mesh for every displacement in this supercell
    kpts_list = kpd
    if isinstance(kpd, int):
        kpts_list = safe_kgrid_from_cell_volume(interchange_atoms_ase_phonopy(phonons.supercell), kpd)

    displacement_directories = [os.path.join(sc_directories[sc_ind], f'disp-{ind+1:03d}') for ind in range(len(supercells))]
    make_directories_from_list(displacement_directories, delete = False)

    force_sets = np.array([get_forces(supercell, displacement_directories[ind], kpts_list)
                           for ind, supercell in enumerate(supercells)])

    # disp-000 is the undisplaced supercell (only run for residual force subtraction)
    if subtract_residual_forces:
        perfect_directory = os.path.join(sc_directories[sc_ind], 'disp-000')
        make_directories_from_list([perfect_directory], delete = False)
        force_sets -= get_forces(phonons.supercell, perfect_directory, kpts_list)

    # Tell phonopy what the forces were. No FC symmetrization here; Quesadilla
    # symmetrizes the final q-space force constants.
    phonons.forces = force_sets
    phonons.produce_force_constants()
    phonons.save(os.path.join(sc_directories[sc_ind], 'phonopy.yaml'), settings = {'force_constants': True})

# ----------------------------------------------------------------------- #
# Quesadilla: D(q) at commensurate q -> unfold stars -> real space FCs on the full grid
ndsc_calc = NondiagonalPhononCalculator(sc_gen, nd_phonons)
ndsc_calc.run()
phonons = ndsc_calc.phonons
phonons.save('phonopy.yaml', settings = {'force_constants': True}) # equivalent to `quesadilla --fc`

# ----------------- #
# Frequencies at the irreducible commensurate q-points. These are the points actually
# sampled by the supercells; anything between them is Fourier interpolation.
phonons.run_qpoints(sc_gen.q_ibz)
frequencies = phonons.get_qpoints_dict()['frequencies']

with open('commensurate_frequencies.dat', 'w') as f:
    f.write('# q (fractional, PPOSCAR reciprocal basis) | frequencies (THz), negative = imaginary\n')
    for q, freqs in zip(sc_gen.q_ibz, frequencies):
        f.write(' '.join(f'{x:9.5f}' for x in q) + ' | ' + ' '.join(f'{x:10.4f}' for x in freqs) + '\n')

print('------------------------------------')
print('Lowest frequency at each irreducible q-point (THz):')
for q, freqs in zip(sc_gen.q_ibz, frequencies):
    print(f'q = {np.round(q, 4).tolist()}: {freqs.min():10.4f}' + ('   <-- imaginary' if freqs.min() < -0.05 else ''))

# ----------------- #
# Band structure and DOS (interpolated)
phonons.auto_band_structure(write_yaml = True, filename = 'band.yaml')
phonons.run_mesh()
phonons.run_total_dos()
phonons.write_total_dos(filename = 'total_dos.dat')
