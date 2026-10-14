# ----------------------------------------------------------------------- #
# Committee NEBs at a set of strains + uncertainty-based frame selection for
# MACE active learning. Run from one path directory (one mechanism).
#
# Inputs, all read from the working directory:
#   Initial.traj, Final.traj           endpoints in the UNSTRAINED cell (built in the notebook)
#   mace_settings.json                 as for Fast_NEB_ASE.py, but 'model_path' may be a LIST:
#                                      the committee. model_path[0] is the reference model.
#   neb_settings.json                  as for Fast_NEB_ASE.py ('restart'/'read_images' ignored --
#                                      restarts and warm starts are handled here)
#   active_learning_settings.json      see below
#
# Usage:  python MACE_Active_Learning.py [stage] [strain_name] [member_index]
#   neb     strained endpoints -> relax (only if that member's model changed) -> NEB.
#           Give a strain name and/or member index to run one (strain, member) per job.
#   select  evaluate every candidate frame with every member, pick frames, write outputs.
#           Run once all 'neb' jobs are done.
#   all     neb for everything, then select (default; fine for small committees/paths).
#
# active_learning_settings.json keys:
#   round              int, labels outputs (selected_images_round<N>.traj). Required.
#   strains            list of dicts, each with a unique 'name' and a 'type':
#                        {"name": "0",        "type": "none"}
#                        {"name": "biax_p2",  "type": "biaxial",  "eps": 0.02}
#                        {"name": "uni_x_m2", "type": "uniaxial", "eps": -0.02, "axis": 0}
#                        {"name": "shear_1",  "type": "shear",    "eps": 0.01, "axes": [0, 1]}
#                        {"name": "rand_0",   "type": "random",   "std": 0.015, "seed": 0}
#                        {"name": "custom",   "type": "tensor",   "tensor": [[...], [...], [...]]}
#                      add "holdout": true to run a strain but never select training frames from it.
#   in_plane_only      true for slabs: only xx, yy, xy may be strained.     default true
#   max_strain         hard cap on any strain component.                    default 0.05
#   relax_fmax         endpoint relaxation, eV/A.                           default 0.005
#   relax_steps        endpoint relaxation step cap.                        default 1000
#   neb_steps          NEB step cap per job (restartable).                  default 2000
#   symmetric          final state equivalent to initial -> skip FS and the mirror-image
#                      TS neighbour (their slots go to uncertainty picks).  default false
#   n_neighbours       images on each side of the TS always taken.         default 1
#   n_extra            uncertainty picks per strain.                        default 1
#   n_rattle           rattled copies of the TS added to the candidates.    default 4
#   rattle_stdev       A.                                                   default 0.03
#   max_rattle_picks   cap on rattled frames among the uncertainty picks -- rattles always
#                      out-score band images, so uncapped they crowd the path out.  default 1
#   energy_scale       eV, sigma(E - E_IS) normaliser in the score.         default 0.005
#   force_scale        eV/A, max per-atom force sigma normaliser.           default 0.05
#   min_displacement   a candidate whose max atomic displacement from an already chosen
#                      frame (this or earlier rounds) is below this is a duplicate.  default 0.05
#   endpoint_tolerance warn when a re-relaxed endpoint moves more than this (max free-atom
#                      displacement, A) from the previous model's endpoint.  default 0.2
#
# Outputs:
#   strain_<name>/member_<k>/   Initial_relaxed.traj, Final_relaxed.traj, NEB.traj, band.traj, state.json
#   strain_<name>/band.traj     reference band, warm start for the next round
#   strain_<name>/selected_history.traj   every frame chosen so far (all rounds), for deduplication
#   selected_images_round<N>.traj         frames for DFT (no MACE labels attached)
#   holdout_images_round<N>.traj          IS/TS/FS of holdout strains, for final validation
#   active_learning_round<N>.json         barriers, committee spreads, picks, warnings
#
## EXAMPLE SETTINGS:
# {"round": 1, "symmetric": true,
#  "strains": [{"name": "0", "type": "none"},
#              {"name": "biax_m2", "type": "biaxial", "eps": -0.02},
#              {"name": "biax_p2", "type": "biaxial", "eps": 0.02},
#              {"name": "uni_x_p2", "type": "uniaxial", "eps": 0.02, "axis": 0},
#              {"name": "rand_0", "type": "random", "std": 0.015, "seed": 0},
#              {"name": "biax_p1", "type": "biaxial", "eps": 0.01, "holdout": true}]}
# ----------------------------------------------------------------------- #

# Imports
    # System
import os
import sys
import json

    # System +
import numpy as np

# ASE Relaxation Stuff
from ase.optimize import BFGS, MDMin, FIRE2, LBFGS
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.geometry import find_mic

    # NEB
from ase.mep.dyneb import DyNEB

    # File io
from ase.io import read, write

#MACE
from mace.calculators import MACECalculator

# ----------------------------------------------------------------------- #
# Read in settings
    # Read NEB Settings
with open('neb_settings.json') as json_file:
    neb_settings = json.load(json_file)
    json_file.close()
neb_settings.pop('restart', None)
neb_settings.pop('read_images', None)
n_images = neb_settings.pop('n_images')
# passed to DyNEB as well as the optimizer -- see the note in Fast_NEB_ASE.py
fmax = neb_settings.pop('fmax', 0.05)
optimizer = neb_settings.pop('optimizer', 'FIRE2')

    # Read MACE Settings
with open('mace_settings.json') as json_file:
    mace_settings = json.load(json_file)
    json_file.close()

model_paths = mace_settings.pop('model_path')
if isinstance(model_paths, str):
    model_paths = [model_paths]
mace_settings.pop('fmax', None)              # not a MACECalculator kwarg

    # Read active learning settings
with open('active_learning_settings.json') as json_file:
    al_settings = json.load(json_file)
    json_file.close()

al_round = al_settings['round']
strains = al_settings['strains']
in_plane_only = al_settings.get('in_plane_only', True)
max_strain = al_settings.get('max_strain', 0.05)
relax_fmax = al_settings.get('relax_fmax', 0.005)
relax_steps = al_settings.get('relax_steps', 1000)
neb_steps = al_settings.get('neb_steps', 2000)
symmetric = al_settings.get('symmetric', False)
n_neighbours = al_settings.get('n_neighbours', 1)
n_extra = al_settings.get('n_extra', 1)
n_rattle = al_settings.get('n_rattle', 4)
rattle_stdev = al_settings.get('rattle_stdev', 0.03)
max_rattle_picks = al_settings.get('max_rattle_picks', 1)
energy_scale = al_settings.get('energy_scale', 0.005)
force_scale = al_settings.get('force_scale', 0.05)
min_displacement = al_settings.get('min_displacement', 0.05)
endpoint_tolerance = al_settings.get('endpoint_tolerance', 0.2)

names = [s['name'] for s in strains]
assert len(set(names)) == len(names), f"strain names must be unique, got {names}"

    # Command line
stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
only_strain = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != 'all' else None
only_member = int(sys.argv[3]) if len(sys.argv) > 3 else None
assert stage in ('neb', 'select', 'all'), f"stage must be neb, select or all, got {stage}"
assert only_strain is None or only_strain in names, f"unknown strain {only_strain}, have {names}"

path_name = os.path.basename(os.getcwd())

# ----------------------------------------------------------------------- #
# Helpers
def strain_tensor(spec):
    """Symmetric small-strain tensor for one entry of 'strains'."""
    kind = spec['type']
    eps = np.zeros((3, 3))
    if kind == 'none':
        pass
    elif kind == 'biaxial':
        eps[0, 0] = eps[1, 1] = spec['eps']
    elif kind == 'uniaxial':
        eps[spec['axis'], spec['axis']] = spec['eps']
    elif kind == 'shear':
        i, j = spec['axes']
        eps[i, j] = eps[j, i] = spec['eps']
    elif kind == 'random':
        rng = np.random.default_rng(spec['seed'])
        comps = [(0, 0), (1, 1), (0, 1)] if in_plane_only else \
                [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        for (i, j) in comps:
            eps[i, j] = eps[j, i] = np.clip(rng.normal(0, spec['std']), -max_strain, max_strain)
    elif kind == 'tensor':
        eps = np.array(spec['tensor'], dtype=float)
        eps = 0.5 * (eps + eps.T)
    else:
        raise ValueError(f"unknown strain type {kind}")

    if in_plane_only:
        assert np.allclose(eps[2, :], 0) and np.allclose(eps[:, 2], 0), (
            f"strain {spec['name']} has out-of-plane components but in_plane_only is true")
    assert np.abs(eps).max() <= max_strain + 1e-12, (
        f"strain {spec['name']} exceeds max_strain = {max_strain}")
    return eps


def apply_strain(atoms, eps):
    strained = atoms.copy()
    strained.set_cell(atoms.cell[:] @ (np.eye(3) + eps), scale_atoms=True)
    return strained


def fixed_mask(atoms):
    mask = np.zeros(len(atoms), dtype=bool)
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            mask[c.index] = True
    return mask


def max_displacement(a, b, free_only=False):
    """Largest minimum-image atomic displacement between two frames (inf if not comparable)."""
    if len(a) != len(b) or (a.numbers != b.numbers).any() or not np.allclose(a.cell[:], b.cell[:], atol=1e-6):
        return np.inf
    d, _ = find_mic(a.positions - b.positions, a.cell, a.pbc)
    norms = np.linalg.norm(d, axis=1)
    if free_only:
        norms = norms[~fixed_mask(a)]
    return float(norms.max())


def fingerprint(path):
    st = os.stat(path)
    return {'path': os.path.abspath(path), 'size': st.st_size, 'mtime': st.st_mtime}


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as json_file:
        return json.load(json_file)


def write_json(path, data):
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=2)


def make_calculator(k):
    return MACECalculator(model_paths=model_paths[k], **mace_settings)


def make_optimizer(obj, trajectory):
    if optimizer == 'BFGS':
        return BFGS(obj, trajectory=trajectory)
    elif optimizer == 'LBFGS':
        return LBFGS(obj, trajectory=trajectory)
    elif optimizer == 'MDMin':
        return MDMin(obj, trajectory=trajectory)
    return FIRE2(obj, trajectory=trajectory, use_abc=True, dt=0.05, maxstep=0.05, dtmax=0.1)


def frozen(atoms, calc):
    """Copy carrying this calculator's energy/forces as a SinglePointCalculator."""
    atoms.calc = calc
    e, f = atoms.get_potential_energy(), atoms.get_forces()
    a = atoms.copy()
    a.calc = SinglePointCalculator(a, energy=e, forces=f)
    return a


def strip_calc(atoms):
    """Copy without MACE results, so they can't be mistaken for DFT labels."""
    a = atoms.copy()          # Atoms.copy() drops the calculator
    for key in ('energy', 'free_energy', 'stress', 'virial'):
        a.info.pop(key, None)
    a.arrays.pop('forces', None)
    return a


def relax(atoms, calc, trajectory):
    atoms = atoms.copy()
    atoms.calc = calc
    BFGS(atoms, trajectory=trajectory, logfile='-').run(fmax=relax_fmax, steps=relax_steps)
    return frozen(atoms, calc)

# ----------------------------------------------------------------------- #
# NEB stage: one (strain, member) at a time
def run_member(spec, k):
    sdir = f"strain_{spec['name']}"
    mdir = os.path.join(sdir, f'member_{k}')
    os.makedirs(mdir, exist_ok=True)
    state_path = os.path.join(mdir, 'state.json')
    state = read_json(state_path, {})
    fp = fingerprint(model_paths[k])
    calc = make_calculator(k)
    eps = strain_tensor(spec)
    tag = f"[strain {spec['name']} | member {k}]"

    # ---------------------------------- #
    # Endpoints: relax only when this member's model changed
    have_endpoints = all(os.path.exists(os.path.join(mdir, f)) for f in ('Initial_relaxed.traj', 'Final_relaxed.traj'))
    if state.get('fingerprint') != fp or not have_endpoints:
        old = [read(os.path.join(mdir, f)) for f in ('Initial_relaxed.traj', 'Final_relaxed.traj')] if have_endpoints else None
        print(f"{tag} model changed or no endpoints -> relaxing endpoints with {model_paths[k]}")
        new = [relax(apply_strain(read(f), eps), calc, os.path.join(mdir, f'relax_{f}'))
               for f in ('Initial.traj', 'Final.traj')]
        drift = [max_displacement(o, n, free_only=True) for o, n in zip(old, new)] if old else [None, None]
        for lbl, d in zip(('initial', 'final'), drift):
            if d is not None and d > endpoint_tolerance:
                print(f"{tag} WARNING: re-relaxed {lbl} endpoint moved {d:.3f} A from the previous model's "
                      f"(> {endpoint_tolerance}) -- check that it is still the same state")
        write(os.path.join(mdir, 'Initial_relaxed.traj'), new[0])
        write(os.path.join(mdir, 'Final_relaxed.traj'), new[1])
        # a band from an older model is only a starting guess now
        if os.path.exists(os.path.join(mdir, 'NEB.traj')):
            os.replace(os.path.join(mdir, 'NEB.traj'), os.path.join(mdir, 'NEB_previous_model.traj'))
        state = {'fingerprint': fp, 'neb_converged': False, 'endpoint_drift': drift}
        write_json(state_path, state)

    if state.get('neb_converged'):
        print(f"{tag} NEB already converged with this model, skipping")
        return

    initial = read(os.path.join(mdir, 'Initial_relaxed.traj'))
    final = read(os.path.join(mdir, 'Final_relaxed.traj'))

    # ---------------------------------- #
    # Starting band: own unfinished NEB > last round's reference band > IDPP
    own = os.path.join(mdir, 'NEB.traj')
    warm = os.path.join(sdir, 'band.traj')
    images = None
    if os.path.exists(own) and os.path.getsize(own) > 0:
        images = read(f'{own}@-{n_images}:')
        start = 'restart from own NEB.traj'
    elif os.path.exists(warm):
        guess = read(f'{warm}@:')
        if len(guess) == n_images and len(guess[0]) == len(initial):
            images = [initial.copy()] + [g.copy() for g in guess[1:-1]] + [final.copy()]
            start = 'warm start from previous band'
    if images is None:
        images = [initial.copy() for _ in range(n_images - 1)] + [final.copy()]
        start = 'IDPP'
    print(f"{tag} NEB starting band: {start}")

    neb = DyNEB(images, fmax=fmax, allow_shared_calculator=True, **neb_settings)
    if start == 'IDPP':
        neb.interpolate(method='idpp')
    for image in neb.images:          # after IDPP, which uses its own calculator
        image.calc = calc

    relaxer = make_optimizer(neb, own)
    converged = bool(relaxer.run(fmax=fmax, steps=neb_steps))

    band = [frozen(image, calc) for image in neb.images]
    write(os.path.join(mdir, 'band.traj'), band)

    energies = np.array([b.get_potential_energy() for b in band])
    state['neb_converged'] = converged
    state['barrier_meV'] = float(1000 * (energies.max() - energies[0]))
    write_json(state_path, state)
    print(f"{tag} converged = {converged}   barrier = {state['barrier_meV']:.1f} meV")

# ----------------------------------------------------------------------- #
# Select stage: committee evaluation of every candidate, then picks
def select_strain(spec, calcs):
    sdir = f"strain_{spec['name']}"
    holdout = spec.get('holdout', False)
    members = [k for k in range(len(model_paths))
               if os.path.exists(os.path.join(sdir, f'member_{k}', 'band.traj'))]
    report = {'holdout': holdout, 'strain_tensor': strain_tensor(spec).tolist(), 'warnings': []}
    if 0 not in members:
        report['warnings'].append('reference member 0 has no band -- strain skipped')
        print(f"[strain {spec['name']}] reference member 0 has no band, skipping")
        return report, [], []
    missing = sorted(set(range(len(model_paths))) - set(members))
    if missing:
        report['warnings'].append(f'members without a band (not in the committee here): {missing}')

    states = {k: read_json(os.path.join(sdir, f'member_{k}', 'state.json'), {}) for k in members}
    for k in members:
        if states[k].get('fingerprint') != fingerprint(model_paths[k]):
            report['warnings'].append(f'member {k} band is from a different/older model -- rerun neb')
        if not states[k].get('neb_converged'):
            report['warnings'].append(f'member {k} NEB not converged')
        for lbl, d in zip(('initial', 'final'), states[k].get('endpoint_drift', [None, None])):
            if d is not None and d > endpoint_tolerance:
                report['warnings'].append(f'member {k} {lbl} endpoint moved {d:.3f} A between models')

    bands = {k: read(os.path.join(sdir, f'member_{k}', 'band.traj@:')) for k in members}
    ref = bands[0]
    e_ref = np.array([b.get_potential_energy() for b in ref])
    ci = int(np.argmax(e_ref))
    interior_max = 0 < ci < len(ref) - 1
    if not interior_max:
        report['warnings'].append('reference band has no interior maximum (barrierless/monotonic) -- no TS picks')

    # ---------------------------------- #
    # Candidates
    cands = [(ref[0], 'IS', 0, 0)]
    if not symmetric:
        cands.append((ref[-1], 'FS', 0, len(ref) - 1))
    for k in members:
        for i in range(1, len(bands[k]) - 1):
            # symmetric path: the second half mirrors the first, which the displacement
            # test can't see -- labelling both would spend DFT on equivalent frames
            if symmetric and i > (len(bands[k]) - 1) // 2:
                continue
            cands.append((bands[k][i], 'band', k, i))
    rng = np.random.default_rng(1000 * al_round + names.index(spec['name']))
    for r in range(n_rattle if interior_max else 0):
        a = ref[ci].copy()
        a.rattle(stdev=rattle_stdev, seed=int(rng.integers(2**31)))   # FixAtoms atoms stay put
        cands.append((a, 'rattle', 0, ci))

    # ---------------------------------- #
    # Committee evaluation, energies relative to each member's view of the reference IS
    m = len(members)
    E = np.zeros((m, len(cands)))
    F = [[None] * len(cands) for _ in range(m)]
    for row, k in enumerate(members):
        e_is = frozen(ref[0].copy(), calcs[k]).get_potential_energy()
        for col, (a, _, _, _) in enumerate(cands):
            b = frozen(a.copy(), calcs[k])
            E[row, col] = b.get_potential_energy() - e_is
            F[row][col] = b.get_forces()

    sigma_dE = E.std(axis=0)
    sigma_F = np.zeros(len(cands))
    for col, (a, _, _, _) in enumerate(cands):
        f = np.array([F[row][col] for row in range(m)])           # (members, atoms, 3)
        per_atom = np.sqrt(((f - f.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))
        per_atom[fixed_mask(a)] = 0.0
        sigma_F[col] = per_atom.max()
    score = sigma_dE / energy_scale + sigma_F / force_scale

    # barrier on each member's own path, and every member's barrier along the reference path
    own_barriers = {k: float(1000 * (max(b.get_potential_energy() for b in bands[k]) - bands[k][0].get_potential_energy()))
                    for k in members}
    ref_cols = [0] + [c for c, (_, kind, k, i) in enumerate(cands) if kind == 'band' and k == 0]
    if not symmetric:
        ref_cols.append(1)
    ref_path_barriers = 1000 * E[:, ref_cols].max(axis=1)
    report.update({
        'members': members,
        'barrier_own_path_meV': own_barriers,
        'barrier_own_path_spread_meV': float(np.ptp(list(own_barriers.values()))),
        'barrier_ref_path_meV': {k: float(b) for k, b in zip(members, ref_path_barriers)},
        'barrier_ref_path_sigma_meV': float(ref_path_barriers.std()),
        'ts_image': ci,
    })

    # ---------------------------------- #
    # Holdout strains: keep IS/TS/FS for final validation only
    if holdout:
        keep = [ref[0], ref[ci], ref[-1]] if interior_max else [ref[0], ref[-1]]
        out = []
        for a, lbl in zip(keep, ('IS', 'TS', 'FS') if interior_max else ('IS', 'FS')):
            a = strip_calc(a)
            a.info.update({'al_path': path_name, 'al_strain': spec['name'], 'al_round': al_round, 'al_reason': f'holdout_{lbl}'})
            out.append(a)
        report['picks'] = []
        return report, [], out

    # ---------------------------------- #
    # Picks: mandatory IS/(FS)/TS/neighbours from the reference band, then by score
    history_path = os.path.join(sdir, 'selected_history.traj')
    history = [h for h in read(f'{history_path}@:')] if os.path.exists(history_path) else []
    history = [h for h in history if h.info.get('al_round') != al_round]     # re-running select is idempotent
    chosen = list(history)

    def is_duplicate(a):
        return any(max_displacement(a, c) < min_displacement for c in chosen)

    mandatory = [('IS', 0)]
    if not symmetric:
        mandatory.append(('FS', len(ref) - 1))
    if interior_max:
        mandatory.append(('TS', ci))
        sides = [-1] if symmetric else [-1, 1]
        for n in range(1, n_neighbours + 1):
            for s in sides:
                if 0 < ci + s * n < len(ref) - 1:
                    mandatory.append((f'TS{s * n:+d}', ci + s * n))
    extra_slots = n_extra + (n_neighbours if symmetric and interior_max else 0)

    def col_of(kind, k, i):
        for c, (_, ck, kk, ii) in enumerate(cands):
            if ck == kind and kk == k and ii == i:
                return c
        return None

    picks, skipped = [], []
    for lbl, i in mandatory:
        col = col_of(lbl, 0, i) if lbl in ('IS', 'FS') else col_of('band', 0, i)
        if is_duplicate(cands[col][0]):
            skipped.append(lbl)
            continue
        picks.append((col, lbl))
        chosen.append(cands[col][0])

    n_rattle_picked = 0
    for col in np.argsort(-score):
        if extra_slots == 0:
            break
        if any(col == p for p, _ in picks) or is_duplicate(cands[col][0]):
            continue
        if cands[col][1] == 'rattle':
            if n_rattle_picked >= max_rattle_picks:
                continue
            n_rattle_picked += 1
        picks.append((int(col), 'uncertainty'))
        chosen.append(cands[col][0])
        extra_slots -= 1

    out = []
    for col, lbl in picks:
        a, kind, k, i = cands[col]
        a = strip_calc(a)
        a.info.update({'al_path': path_name, 'al_strain': spec['name'], 'al_round': al_round,
                       'al_reason': lbl, 'al_source': kind, 'al_member': k, 'al_image': i,
                       'al_sigma_dE_meV': float(1000 * sigma_dE[col]),
                       'al_sigma_F_meVA': float(1000 * sigma_F[col])})
        out.append(a)

    write(history_path, [strip_calc(h) for h in history] + out)
    write(os.path.join(sdir, 'band.traj'), ref)        # warm start for the next round
    report['picks'] = [{k: v for k, v in a.info.items() if k.startswith('al_')} for a in out]
    report['skipped_as_duplicate'] = skipped
    return report, out, []

# ----------------------------------------------------------------------- #
# Run
if stage in ('neb', 'all'):
    for spec in strains:
        if only_strain is not None and spec['name'] != only_strain:
            continue
        for k in range(len(model_paths)):
            if only_member is not None and k != only_member:
                continue
            run_member(spec, k)

if stage in ('select', 'all'):
    calcs = {k: make_calculator(k) for k in range(len(model_paths))}
    summary = {'path': path_name, 'round': al_round, 'models': model_paths, 'strains': {}}
    selected, holdout_frames = [], []
    for spec in strains:
        report, out, held = select_strain(spec, calcs)
        summary['strains'][spec['name']] = report
        selected += out
        holdout_frames += held

    write(f'selected_images_round{al_round}.traj', selected)
    if holdout_frames:
        write(f'holdout_images_round{al_round}.traj', holdout_frames)
    write_json(f'active_learning_round{al_round}.json', summary)

    # ---------------------------------- #
    # Table for the log
    print(f"\n{path_name}  round {al_round}  committee of {len(model_paths)}")
    print(f"{'strain':>12} {'Ea ref (meV)':>13} {'sigma ref':>10} {'spread own':>11} {'picks':>6}  warnings")
    for name, r in summary['strains'].items():
        if 'barrier_ref_path_meV' not in r:
            print(f"{name:>12} {'--':>13} {'--':>10} {'--':>11} {'--':>6}  {'; '.join(r['warnings'])}")
            continue
        ea = r['barrier_ref_path_meV'][0]
        n = 'hold' if r['holdout'] else len(r['picks'])
        print(f"{name:>12} {ea:13.1f} {r['barrier_ref_path_sigma_meV']:10.1f} "
              f"{r['barrier_own_path_spread_meV']:11.1f} {n:>6}  {'; '.join(r['warnings'])}")
    print(f"\nwrote selected_images_round{al_round}.traj ({len(selected)} frames)")
