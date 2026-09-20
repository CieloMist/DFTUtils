"""
Generate symmetry-inequivalent interstitial H sites with `doped`, returned as ASE Atoms.

Bulk (doped builds a supercell):

    from generate_h_interstitials import generate_h_interstitials, kmesh
    bulk, defects = generate_h_interstitials("Beta_Tin/Sn(1).cif")

Slab (keep the cell you built, only search inside it):

    bulk, defects = generate_h_interstitials(my_slab, generate_supercell=False)

NOTE: doped uses multiprocessing.  `processes=1` is the default here so this is safe
to call from a Jupyter notebook or an unguarded script.  If you set processes>1 you
must call it from inside an `if __name__ == "__main__":` block.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import find_optimal_cell_shape, make_supercell
from ase.io import read
from doped.generation import DefectsGenerator
from pymatgen.io.ase import AseAtomsAdaptor

# 2*pi / (n*L), with n*L = 94 A from the slab k-point convergence study
KSPACING = 0.0668  # Angstrom^-1


def _to_ase(structure) -> Atoms:
    """pymatgen Structure -> ASE Atoms (oxidation states and dangling site props stripped).

    Site properties survive an ASE -> pymatgen -> ASE round trip.  If the input Atoms
    carried tags (`ase.build.surface()` sets them to the layer index), the inserted
    defect site has no tag, so pymatgen hands `ase.Atoms.set_tags()` a list containing
    None and it raises `TypeError: int() argument ... not 'NoneType'`.  Backfill
    numeric properties with 0 -- which keeps your layer tags usable for FixAtoms and
    gives the interstitial tag 0 -- and drop non-numeric ones.
    """
    s = structure.copy()
    s.remove_oxidation_states()
    for prop, vals in list(s.site_properties.items()):
        if not any(v is None for v in vals):
            continue
        present = [v for v in vals if v is not None]
        if present and all(isinstance(v, (bool, int, float, np.integer, np.floating))
                           for v in present):
            s.add_site_property(prop, [0 if v is None else v for v in vals])
        else:
            s.remove_site_property(prop)
    return AseAtomsAdaptor.get_atoms(s)


def vacuum_thickness(atoms: Atoms) -> np.ndarray:
    """Vacuum thickness (Angstrom) along each of the three cell axes."""
    b = np.linalg.norm(2 * np.pi * np.array(atoms.cell.reciprocal()), axis=1)
    d_perp = 2 * np.pi / b                     # interplanar spacing along each axis
    frac = atoms.get_scaled_positions() % 1.0
    out = np.zeros(3)
    for i in range(3):
        f = np.sort(frac[:, i])
        gaps = np.diff(np.concatenate([f, [f[0] + 1.0]]))
        out[i] = gaps.max() * d_perp[i]
    return out


def kmesh(atoms: Atoms, kspacing: float = KSPACING, vacuum_tol: float = 8.0) -> list[int]:
    """Gamma-centred mesh at fixed reciprocal-space density.

    Works for skewed cells, where `n * L` is meaningless -- doped's supercells are
    minimum-atom-count and often strongly non-orthogonal.

    Any axis carrying more than `vacuum_tol` of vacuum gets n = 1: sampling
    dispersion across a vacuum gap is wasted effort, and a pure reciprocal-density
    rule would silently ask for 2-3 k-points there.
    """
    b = np.linalg.norm(2 * np.pi * np.array(atoms.cell.reciprocal()), axis=1)
    vac = vacuum_thickness(atoms)
    return [1 if vac[i] > vacuum_tol else max(1, int(np.ceil(b[i] / kspacing)))
            for i in range(3)]



def _interstitial_sites_frac(host: Atoms, all_wyckoffs=True, include_adsorbates=False):
    """Fractional coordinates of the symmetry-inequivalent interstitial sites of `host`.

    Runs doped's Voronoi analysis on the *input* cell (cheap) rather than on a
    large supercell, and returns one representative per inequivalent site.
    """
    from doped.generation import get_interstitial_sites
    kw = {"include_unique_wyckoffs": all_wyckoffs}
    if not include_adsorbates:
        kw["vacuum_radius"] = 99.0
    pmg = AseAtomsAdaptor.get_structure(host)
    out = []
    for entry in get_interstitial_sites(pmg, **kw):
        fc = entry[0] if isinstance(entry, (tuple, list)) else entry
        mult = entry[1] if isinstance(entry, (tuple, list)) and len(entry) > 1 else 1
        eq = entry[2] if isinstance(entry, (tuple, list)) and len(entry) > 2 else [fc]
        out.append((np.asarray(fc, dtype=float), int(mult),
                    [np.asarray(e, dtype=float) for e in eq]))
    return out


def _supercell_by_target_size(host: Atoms, target_size: int, target_shape: str = "sc",
                              size_tolerance: float = 0.0, **shape_kwargs):
    """Compact supercell of ~`target_size` copies of `host`, via ase.build.

    `target_shape='sc'` drives the cell toward cubic, which keeps |b_i| small
    and therefore the k-point count down -- a skewed cell of equal volume can
    need ~1.5x more k-points.

    ase's find_optimal_cell_shape takes an EXACT size, so `size_tolerance`
    is implemented here by scanning sizes in
    [target_size, ceil(target_size * (1 + size_tolerance))] and keeping the one
    with the smallest shape deviation -- i.e. spend extra atoms to buy
    orthogonality.  This is the analogue of doped's `ideal_threshold`, which
    only applies to the min_image_distance route.

    `shape_kwargs` are forwarded to find_optimal_cell_shape
    (lower_limit, upper_limit, score_key, verbose).

    Returns (supercell, P, size_used, deviation).
    """
    from ase.build import get_deviation_from_optimal_cell_shape as _dev
    hi = int(np.ceil(target_size * (1.0 + max(0.0, size_tolerance))))
    best = None
    for n in range(int(target_size), hi + 1):
        try:
            P = np.asarray(find_optimal_cell_shape(host.cell, n, target_shape,
                                                   **shape_kwargs), dtype=int)
        except Exception:
            continue
        if abs(int(round(np.linalg.det(P)))) != n:
            continue
        cell = P @ np.asarray(host.cell)
        try:
            score = float(_dev(cell, target_shape=target_shape))
        except Exception:
            L = np.linalg.norm(cell, axis=1)
            score = float(L.max() / L.min() - 1.0)
        if best is None or score < best[3]:
            best = (make_supercell(host, P), P, n, score)
    if best is None:
        raise RuntimeError(f"no valid supercell found for target_size={target_size}")
    return best

def generate_h_interstitials(
    host,
    generate_supercell: bool = True,
    min_image_distance: float = 13.0,
    min_atoms: int = 30,
    target_size: int | None = None,
    target_shape: str = "sc",
    size_tolerance: float = 0.0,
    shape_kwargs: dict | None = None,
    all_wyckoffs: bool = True,
    charge_states=(0,),
    include_adsorbates: bool = False,
    max_host_distance: float = 4.0,
    processes: int = 1,
):
    """
    Parameters
    ----------
    host : ase.Atoms, pymatgen Structure, or path to a structure file.
    generate_supercell : if False, use `host` exactly as given (cell preserved) and
        only search for interstitial sites inside it.  Use this for a slab you built.
    min_image_distance : min distance between periodic images of the defect (A).
        Only used when generate_supercell=True.  13 A clears MACE's receptive field
        (~2 layers x 6 A), so the local environments are dilute-limit.
    min_atoms : minimum atoms in the generated supercell (generate_supercell=True).
    target_size : if given, build the supercell as `target_size` copies of `host`
        using ase.build.find_optimal_cell_shape, and place one H at each
        inequivalent interstitial site.  Overrides generate_supercell /
        min_image_distance.  Use this to dial H concentration: 1 H per
        target_size cells.
    target_shape : 'sc' (drive toward cubic, default) or 'fcc'.  Compact cells
        need fewer k-points than skewed ones of equal volume.
    size_tolerance : fraction of EXTRA cells allowed above target_size in
        exchange for a more orthogonal cell, e.g. 0.25 searches
        target_size..1.25*target_size and keeps the best-shaped result.
        ase's find_optimal_cell_shape has no such knob; this scans for you.
        (doped's `ideal_threshold` does the same job on the
        min_image_distance route.)
    shape_kwargs : forwarded to ase.build.find_optimal_cell_shape --
        lower_limit / upper_limit (search range for P, default -2..2),
        score_key, verbose.
    all_wyckoffs : keep every symmetry-inequivalent Voronoi site.  doped's default
        (False) returns only the single highest-symmetry site.
    charge_states : charge states to keep.  (0,) for MLIP training; None keeps all.
    include_adsorbates : if the structure has vacuum, doped also proposes adsorbate
        sites above the surface.  False suppresses them (interstitials only).
    max_host_distance : drop any site whose nearest host atom is further than this.
        Removes spurious Voronoi sites in the middle of a vacuum gap.
    processes : doped multiprocessing.  Leave at 1 unless you wrap the call in
        `if __name__ == "__main__":`.

    Returns
    -------
    bulk : ase.Atoms                pristine reference cell.
    defects : dict[str, ase.Atoms]  keyed by doped's defect name.  Each carries
        `name`, `charge`, `h_frac_coords`, `h_height` and `nearest_host` in
        `atoms.info`.
    """
    if isinstance(host, str):
        host = read(host)

    # ---- target_size route: build a compact supercell, then place H ----------
    if target_size is not None:
        sites = _interstitial_sites_frac(host, all_wyckoffs, include_adsorbates)
        sc, P, size_used, deviation = _supercell_by_target_size(
            host, target_size, target_shape, size_tolerance, **(shape_kwargs or {}))
        Pinv = np.linalg.inv(P)
        bulk = sc.copy()
        defects = {}
        for i, (frac_prim, mult, _eq) in enumerate(sites):
            frac_sc = (frac_prim @ Pinv) % 1.0        # r = f.h_p ; f_sc = f.P^-1
            atoms = sc.copy()
            atoms.append(Atoms("H", scaled_positions=[frac_sc], cell=sc.cell)[0])
            h = len(atoms) - 1
            others = [j for j in range(len(atoms)) if j != h]
            nearest = float(atoms.get_distances(h, others, mic=True).min())
            if nearest > max_host_distance:
                continue
            name = f"H_i_s{i}_Sn{nearest:.2f}"
            atoms.info.update(name=name, charge=0, nearest_host=nearest,
                              h_height=float(atoms.positions[h, 2]),
                              h_frac_coords=tuple(float(x) for x in frac_sc),
                              multiplicity=mult, target_size=size_used,
                              shape_deviation=deviation,
                              supercell_matrix=P.tolist())
            defects[name] = atoms
        return bulk, defects

    interstitial_kwargs = {"include_unique_wyckoffs": all_wyckoffs}
    if not include_adsorbates:
        interstitial_kwargs["vacuum_radius"] = 99.0  # never classify as a slab

    kwargs = dict(extrinsic="H", generate_supercell=generate_supercell,
                  processes=processes, interstitial_gen_kwargs=interstitial_kwargs)
    if generate_supercell:
        kwargs["supercell_gen_kwargs"] = {"min_image_distance": min_image_distance,
                                          "min_atoms": min_atoms}

    dg = DefectsGenerator(host, **kwargs)

    bulk, defects = None, {}
    for name, entry in dg.defect_entries.items():
        if not name.startswith("H_i"):
            continue
        if charge_states is not None and entry.charge_state not in charge_states:
            continue

        atoms = _to_ase(entry.defect_supercell)
        h = atoms.get_chemical_symbols().index("H")
        others = [i for i in range(len(atoms)) if i != h]
        nearest = float(atoms.get_distances(h, others, mic=True).min())
        if nearest > max_host_distance:
            continue  # site sits in vacuum, not in the material

        atoms.info.update(name=name, charge=entry.charge_state,
                          nearest_host=nearest, h_height=float(atoms.positions[h, 2]))
        site = getattr(entry, "defect_supercell_site", None)
        if site is not None:
            atoms.info["h_frac_coords"] = tuple(float(x) for x in site.frac_coords)
        defects[name] = atoms

        if bulk is None:
            bulk = _to_ase(entry.bulk_supercell)

    return bulk, defects


if __name__ == "__main__":
    import sys
    from ase.io import write

    src = sys.argv[1] if len(sys.argv) > 1 else "Beta_Tin/Sn(1).cif"
    slab_mode = "--slab" in sys.argv
    bulk, defects = generate_h_interstitials(src, generate_supercell=not slab_mode)

    print(f"\nreference cell : {bulk.get_chemical_formula()}  ({len(bulk)} atoms)")
    print(f"cell lengths   : {bulk.cell.lengths().round(3)}")
    print(f"cell angles    : {bulk.cell.angles().round(1)}")
    print(f"vacuum / axis  : {vacuum_thickness(bulk).round(2)}")
    print(f"suggested kmesh: {kmesh(bulk)}\n")
    write("bulk_ref.traj", bulk)

    for name, at in defects.items():
        print(f"  {name:30s} {at.get_chemical_formula():8s} "
              f"z={at.info['h_height']:6.2f}  nearest={at.info['nearest_host']:5.3f} A  "
              f"kmesh={kmesh(at)}")
        write(f"{name}.traj", at)
    print(f"\nwrote {len(defects) + 1} .traj files")


def h_site_grid(host: Atoms, target_size: int = 32, target_shape: str = "sc",
                size_tolerance: float = 0.0, all_wyckoffs: bool = True,
                supercell_matrix=None, dedup_tol: float = 0.05,
                **shape_kwargs):
    """All candidate interstitial H positions in a compact supercell.

    Returns (supercell, positions) where `positions` is a list of
    (cartesian_xyz, site_index) for every symmetry-equivalent interstitial
    position of every inequivalent site type, tiled through the supercell.
    Use it to build multi-H configurations at chosen H-H separations.

    `target_size` counts copies of `host`, not atoms: a 2-atom primitive
    beta-Sn with target_size=32 and the 4-atom conventional cell with
    target_size=16 both give Sn64.  The site types (site_index) are the
    entries of _interstitial_sites_frac(host); sc.info["h_site_types"] lists
    them with their nearest-Sn distance.

    supercell_matrix : explicit P (3x3, or 3 diagonal entries) instead of
        target_size.  find_optimal_cell_shape only searches entries -2..2, so
        e.g. the orthogonal Sn64 cell of conventional beta-Sn, diag(2,2,4),
        is only reachable this way.
    dedup_tol : grid points closer than this (A) are the same site.
    """
    from ase.geometry import get_distances
    sites = _interstitial_sites_frac(host, all_wyckoffs=all_wyckoffs)
    if supercell_matrix is not None:
        P = np.asarray(supercell_matrix, dtype=int)
        if P.ndim == 1:
            P = np.diag(P)
        sc, size_used, dev = make_supercell(host, P), int(round(abs(np.linalg.det(P)))), None
    else:
        sc, P, size_used, dev = _supercell_by_target_size(
            host, target_size, target_shape, size_tolerance, **shape_kwargs)
    Pinv = np.linalg.inv(P)
    n = int(round(abs(np.linalg.det(P))))
    r = int(np.ceil(n ** (1 / 3))) + 2
    seen, cand = set(), []
    for idx, (_fc, _mult, eq) in enumerate(sites):
        for base in eq:
            for t in np.ndindex(2 * r, 2 * r, 2 * r):
                f = ((base + np.array(t) - r) @ Pinv) % 1.0
                key = tuple(np.round(f, 4) % 1.0)      # cheap first pass
                if key not in seen:
                    seen.add(key)
                    cand.append((f @ np.asarray(sc.cell), idx))

    # Exact pass by distance: rounding keys alone miss sites whose fractional
    # coordinate sits on a rounding boundary (e.g. 0.3055/4 = 0.076375).
    xyz = np.array([p for p, _ in cand])
    types = np.array([s for _, s in cand])
    _, D = get_distances(xyz, cell=sc.cell, pbc=sc.pbc)
    keep = np.ones(len(xyz), bool)
    for i in range(len(xyz)):
        if keep[i]:
            keep[i + 1:] &= D[i, i + 1:] > dedup_tol
    xyz, types = xyz[keep], types[keep]
    out = [(p, int(s)) for p, s in zip(xyz, types)]

    _, dh = get_distances(xyz, sc.positions, cell=sc.cell, pbc=sc.pbc)
    sc.info.update(target_size=size_used, shape_deviation=dev,
                   supercell_matrix=P.tolist(),
                   h_site_types={int(s): {"count": int((types == s).sum()),
                                          "nearest_host": round(float(dh[types == s].min()), 3)}
                                 for s in np.unique(types)})
    return sc, out


def make_h_configuration(sc: Atoms, positions, n_h: int, *,
                         target_separation=None, min_separation: float = 1.6,
                         site_types=None, tol: float = 0.05,
                         rng=None, max_tries: int = 200):
    """Place `n_h` H atoms on the candidate grid from h_site_grid().

    target_separation : for n_h == 2, choose a pair whose minimum-image H-H
        distance is within `tol` of the closest achievable value (Angstrom);
        ties (e.g. the same distance along a vs c) are broken at random, so
        repeated calls with different `rng` give different pair orientations.
        Ignored for other n_h.  Beyond about half the shortest cell width the
        partner's periodic images sit at comparable distances, so the "pair"
        is really an ordered H sublattice, not an isolated pair.
    min_separation : no H-H pair closer than this.
    site_types : restrict to these site indices (see sc.info["h_site_types"]),
        e.g. [0] to use only the most open site.  None = all.
    rng : np.random.Generator or int seed, for reproducible configurations.
    """
    from ase.geometry import get_distances
    rng = np.random.default_rng(rng)
    xyz = np.array([p for p, _ in positions])
    types = np.array([s for _, s in positions])
    if site_types is not None:
        keep = np.isin(types, list(site_types))
        xyz, types = xyz[keep], types[keep]
    if len(xyz) < n_h:
        raise ValueError(f"only {len(xyz)} candidate sites for n_h={n_h}")

    def build(pick):
        atoms = sc.copy()
        atoms += Atoms("H" * len(pick), positions=xyz[pick])
        atoms.info.update(n_h=int(n_h), at_pct_H=100.0 * n_h / len(atoms),
                          h_site_idx=[int(types[k]) for k in pick])
        return atoms

    if n_h == 2 and target_separation is not None:
        _, D = get_distances(xyz, cell=sc.cell, pbc=sc.pbc)
        iu, ju = np.triu_indices(len(xyz), k=1)
        d = D[iu, ju]
        ok = d >= min_separation
        if not ok.any():
            raise RuntimeError("no H pair satisfies min_separation")
        score = np.where(ok, np.abs(d - target_separation), np.inf)
        cand = np.flatnonzero(score <= score.min() + tol)
        c = rng.choice(cand)
        if score[c] > 0.1:
            import warnings
            warnings.warn(f"closest achievable H-H distance is {d[c]:.3f} A "
                          f"(asked {target_separation} A)")
        atoms = build([iu[c], ju[c]])
        atoms.info["hh_distance"] = float(d[c])
        return atoms

    # random sequential placement: each new H must clear every H already placed
    for _ in range(max_tries):
        order = rng.permutation(len(xyz))
        pick = [order[0]]
        for k in order[1:]:
            if len(pick) == n_h:
                break
            _, dk = get_distances(xyz[k], xyz[pick], cell=sc.cell, pbc=sc.pbc)
            if dk.min() >= min_separation:
                pick.append(k)
        if len(pick) < n_h:
            continue
        atoms = build(pick)
        if n_h > 1:
            _, D = get_distances(xyz[pick], cell=sc.cell, pbc=sc.pbc)
            atoms.info["hh_min_distance"] = float(D[np.triu_indices(n_h, 1)].min())
        return atoms
    raise RuntimeError(f"could not place {n_h} H with min_separation={min_separation}")
