"""
interface_matcher.py
====================

Implementation of the surface/interface matching algorithm described in

    D. Stradi, L. Jelver, S. Smidstrup, K. Stokbro,
    "Method for determining optimal supercell representation of interfaces",
    J. Phys.: Condens. Matter 29 (2017) 185901.

Given two surfaces A and B (ASE slabs whose surface normal is along z), the
algorithm searches for commensurate interface supercells with strain and
relative rotation below user-defined thresholds. It follows the paper's four
steps:

    1. Build trial surface supercells A* and B* from the two surface unit
       cells using 2x2 integer matrices N and M  (eqs 1-3).
    2. Rotationally align B* so that u1 is parallel to v1            (eq 4).
    3. Match the supercells with a strain tensor eps applied to B*   (eqs 5-7).
    4. Accept the interface supercell if strain / rotation are below
       the thresholds                                               (eq 8).

The geometric search (steps 1-4) is pure NumPy and has no ASE dependency, so it
can be tested/used on its own. `build_interface` then turns an accepted match
into a real atomic structure using ASE.

Implementation note on step 1 / "exclude equivalent lattices"
-------------------------------------------------------------
Rather than looping over every integer matrix N with |n_ij| <= Nmax (which
generates the same sub-lattice many times over), we enumerate each *distinct*
sub-lattice exactly once through its Hermite Normal Form

        H = [[a, b],
             [0, c]] ,   a*c = n,   0 <= b < c ,   a, c > 0

where n = det(H) is the supercell area expressed in units of the surface unit
cell area. This is the standard Zur-McGill construction that the Stradi paper
builds on; it is what makes the search scale ~ Nmax^2 * N_phi as quoted in the
paper, and it automatically satisfies "we exclude equivalent lattices".
The cap on n is derived from the user's `max_area` (or given directly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Small 2x2 geometry helpers
# ----------------------------------------------------------------------------
def _rotation(theta: float) -> np.ndarray:
    """2x2 rotation matrix R(theta)  (paper eq 4)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _angle(vec: np.ndarray) -> float:
    """Polar angle of a 2D vector w.r.t. +x."""
    return np.arctan2(vec[1], vec[0])


def surface_lattice(atoms) -> np.ndarray:
    """
    Return the 2x2 in-plane lattice of an ASE slab as rows (a1, a2).

    The slab is assumed oriented with its surface normal along z, i.e. the
    first two cell vectors lie in the xy-plane.
    """
    cell = np.asarray(atoms.cell)
    a1, a2 = cell[0, :2], cell[1, :2]
    if abs(cell[0, 2]) > 1e-6 or abs(cell[1, 2]) > 1e-6:
        raise ValueError(
            "Surface cell vectors are not in the xy-plane. Orient the slab so "
            "its surface normal points along z (e.g. via ase.build.surface)."
        )
    return np.array([a1, a2], dtype=float)


# ----------------------------------------------------------------------------
# Step 1: enumeration of distinct sub-lattices (supercell matrices)
# ----------------------------------------------------------------------------
def _hnf_matrices(n: int) -> List[np.ndarray]:
    """
    All Hermite-Normal-Form 2x2 integer matrices of determinant n.

    These enumerate the sub-lattices of index n exactly once (no duplicates),
    realising step 1 of the algorithm while "excluding equivalent lattices".
    """
    mats = []
    for a in range(1, n + 1):
        if n % a:
            continue
        c = n // a
        for b in range(c):
            mats.append(np.array([[a, b], [0, c]], dtype=int))
    return mats


def _supercell_matrices(max_index: int) -> List[np.ndarray]:
    """All distinct supercell matrices with area-index 1 .. max_index."""
    out = []
    for n in range(1, max_index + 1):
        out.extend(_hnf_matrices(n))
    return out


# ----------------------------------------------------------------------------
# Steps 2 & 3: align (eq 4) and compute the strain tensor (eqs 5-7)
# ----------------------------------------------------------------------------
def _strain_tensor(V: np.ndarray, U: np.ndarray):
    """
    Align B*'s lattice U to A*'s lattice V and return (eps, theta).

    Parameters
    ----------
    V : (2,2) rows v1, v2  -- supercell of A (= N @ a)
    U : (2,2) rows u1, u2  -- supercell of B (= M @ b)

    Returns
    -------
    eps   : (2,2) symmetric strain tensor [[eps_xx, eps_xy],[eps_xy, eps_yy]]
    theta : rotation applied to B* to align u1 with v1 (radians, eq 4)

    The strain components are evaluated in the canonical frame where v1 lies on
    +x and u1 (after the alignment rotation) also lies on +x. Instead of
    building rotation matrices, the canonical-frame components are obtained
    directly by projecting the second vector onto the (parallel, perpendicular)
    directions of the first: for a first vector w1 and second vector w2,
        w1x = |w1|,  w2x = (w2 . w1)/|w1|,  w2y = (w1 x w2)/|w1| .
    This is identical to eqs (5)-(7) but avoids per-pair matrix algebra, which
    matters in the inner search loop.
    """
    v1, v2 = V
    u1, u2 = U

    nv = np.hypot(v1[0], v1[1])          # |v1|
    nu = np.hypot(u1[0], u1[1])          # |u1|

    # Canonical-frame components (v1 on +x, u1 on +x).
    v2x = (v2[0] * v1[0] + v2[1] * v1[1]) / nv          # v2 . v1hat
    v2y = (v1[0] * v2[1] - v1[1] * v2[0]) / nv          # v1 x v2
    u2x = (u2[0] * u1[0] + u2[1] * u1[1]) / nu          # u2 . u1hat
    u2y = (u1[0] * u2[1] - u1[1] * u2[0]) / nu          # u1 x u2

    # Step 2 -- rotation that aligns u1 with v1 (eq 4).
    theta = np.arctan2(v1[1], v1[0]) - np.arctan2(u1[1], u1[0])

    # Step 3 -- strain components (paper eqs 5, 6, 7).
    #
    # Note: this is the *symmetric* strain tensor (mind the 1/2 in eq 7). The
    # exact deformation D that maps U onto V after alignment is generally
    # non-symmetric (D_yx = 0 because u1 has been aligned with v1, and
    # eps_xy = D_xy/2). eps is the physically meaningful strain that is reported
    # and thresholded here; `build_interface` applies the exact deformation D
    # to the atoms (via set_cell + scale_atoms), so no information is lost.
    eps_xx = nv / nu - 1.0
    eps_yy = v2y / u2y - 1.0
    eps_xy = 0.5 * (v2x - (nv / nu) * u2x) / u2y

    eps = np.array([[eps_xx, eps_xy], [eps_xy, eps_yy]])
    return eps, theta


# ----------------------------------------------------------------------------
# Container for an accepted match
# ----------------------------------------------------------------------------
@dataclass
class InterfaceMatch:
    """A single accepted interface supercell (one solution of eq 8)."""
    N: np.ndarray            # 2x2 integer supercell matrix for surface A
    M: np.ndarray            # 2x2 integer supercell matrix for surface B
    theta: float             # rotation of B* aligning u1->v1 (radians)
    strain: np.ndarray       # 2x2 symmetric strain tensor eps
    area: float              # area of the interface cell (Angstrom^2)
    n_atoms_A: int           # atoms contributed by A (= |det N| * len(A))
    n_atoms_B: int           # atoms contributed by B (= |det M| * len(B))

    @property
    def theta_deg(self) -> float:
        return np.degrees(self.theta)

    @property
    def max_abs_strain(self) -> float:
        """max(|eps_xx|, |eps_yy|, |eps_xy|) -- the per-component criterion."""
        return float(np.max(np.abs([self.strain[0, 0],
                                    self.strain[1, 1],
                                    self.strain[0, 1]])))

    @property
    def rms_strain(self) -> float:
        return float(np.sqrt(np.mean(self.strain ** 2)))

    @property
    def n_atoms(self) -> int:
        return self.n_atoms_A + self.n_atoms_B

    def __repr__(self) -> str:
        e = self.strain
        return (f"InterfaceMatch(area={self.area:7.2f} A^2, atoms={self.n_atoms:3d}, "
                f"theta={self.theta_deg:6.2f} deg, "
                f"eps_xx={e[0,0]:+.4f}, eps_yy={e[1,1]:+.4f}, eps_xy={e[0,1]:+.4f}, "
                f"max|eps|={self.max_abs_strain*100:5.2f}%)")


# ----------------------------------------------------------------------------
# Steps 1-4 combined: the search
# ----------------------------------------------------------------------------
def find_interfaces(
    surface_A,
    surface_B,
    max_strain: float = 0.05,
    max_area: float = 200.0,
    max_angle: Optional[float] = None,
    strain_B: bool = True,
    tol: float = 1e-9,
) -> List[InterfaceMatch]:
    """
    Search for interface supercells matching surface A and surface B.

    Parameters
    ----------
    surface_A, surface_B
        ASE ``Atoms`` slabs (normal along z), OR plain 2x2 arrays giving the
        in-plane lattices (rows a1,a2 / b1,b2) if you only want the geometry.
    max_strain : float
        Threshold on each strain component |eps_xx|,|eps_yy|,|eps_xy| (eq 8,
        condition ii). 0.05 = 5 %.
    max_area : float
        Largest allowed interface-cell area in Angstrom^2. Together with the
        unit-cell areas this fixes the area-index cap (the paper's Nmax/Mmax,
        condition i).
    max_angle : float or None
        Optional threshold on the relative rotation |theta| in **degrees**
        (eq 8, condition iii). ``None`` allows all rotations.
    strain_B : bool
        If True, B is strained onto A (paper's default). If False, A is
        strained onto B. (Straining both equally is a trivial variant.)
    tol : float
        Numerical tolerance for areas / degeneracy.

    Returns
    -------
    list[InterfaceMatch]
        Accepted matches, sorted by (max|strain|, area).
    """
    # Accept either Atoms objects or bare 2x2 lattices.
    L_A = surface_lattice(surface_A) if hasattr(surface_A, "cell") else np.asarray(surface_A, float)
    L_B = surface_lattice(surface_B) if hasattr(surface_B, "cell") else np.asarray(surface_B, float)
    nat_A = len(surface_A) if hasattr(surface_A, "cell") else 1
    nat_B = len(surface_B) if hasattr(surface_B, "cell") else 1

    area_A = abs(np.linalg.det(L_A))
    area_B = abs(np.linalg.det(L_B))

    # Condition (i): area-index caps derived from max_area.
    max_index_A = max(1, int(np.floor(max_area / area_A + tol)))
    max_index_B = max(1, int(np.floor(max_area / area_B + tol)))

    sup_A = _supercell_matrices(max_index_A)
    sup_B = _supercell_matrices(max_index_B)

    # Pre-compute B supercells, sort by area, and cache the canonical-frame
    # quantities so the inner loop can be vectorised over all area-compatible
    # B cells at once (no per-pair Python overhead). Area must match within
    # ~ the strain budget: |dArea/Area| ~ eps_xx + eps_yy.
    import bisect

    area_window = (1.0 + max_strain) ** 2 - 1.0
    lo, hi = 1.0 - area_window - tol, 1.0 + area_window + tol

    B_list = []
    for M in sup_B:
        U = M @ L_B
        u1, u2 = U
        nu = np.hypot(u1[0], u1[1])
        u2x = (u2[0] * u1[0] + u2[1] * u1[1]) / nu
        u2y = (u1[0] * u2[1] - u1[1] * u2[0]) / nu
        phiu = np.arctan2(u1[1], u1[0])
        B_list.append((abs(np.linalg.det(U)), M, nu, u2x, u2y, phiu))
    B_list.sort(key=lambda t: t[0])

    B_area = np.array([t[0] for t in B_list])
    B_M = [t[1] for t in B_list]
    B_nu = np.array([t[2] for t in B_list])
    B_u2x = np.array([t[3] for t in B_list])
    B_u2y = np.array([t[4] for t in B_list])
    B_phiu = np.array([t[5] for t in B_list])
    B_area_sorted = list(B_area)

    matches: List[InterfaceMatch] = []
    seen = set()
    ang = np.inf if max_angle is None else max_angle

    for N in sup_A:
        V = N @ L_A
        v1, v2 = V
        nv = np.hypot(v1[0], v1[1])
        v2x = (v2[0] * v1[0] + v2[1] * v1[1]) / nv
        v2y = (v1[0] * v2[1] - v1[1] * v2[0]) / nv
        phiv = np.arctan2(v1[1], v1[0])
        areaV = abs(np.linalg.det(V))

        # Slice of B cells with compatible area: areaU in [areaV/hi, areaV/lo].
        i0 = bisect.bisect_left(B_area_sorted, areaV / hi)
        i1 = bisect.bisect_right(B_area_sorted, areaV / lo)
        if i1 <= i0:
            continue

        nu = B_nu[i0:i1]
        u2x = B_u2x[i0:i1]
        u2y = B_u2y[i0:i1]
        phiu = B_phiu[i0:i1]

        ratio = nv / nu                                  # = 1 + eps_xx
        if strain_B:
            exx = ratio - 1.0
            eyy = v2y / u2y - 1.0
            exy = 0.5 * (v2x - ratio * u2x) / u2y
            theta = phiv - phiu
        else:
            # Strain A onto B: swap roles (V<->U) component-wise.
            inv = nu / nv                                # |u1|/|v1|
            exx = inv - 1.0
            eyy = u2y / v2y - 1.0
            exy = 0.5 * (u2x - inv * v2x) / v2y
            theta = -(phiv - phiu)

        # Threshold masks (conditions ii and iii of eq 8).
        good = ((np.abs(exx) <= max_strain) &
                (np.abs(eyy) <= max_strain) &
                (np.abs(exy) <= max_strain) &
                (np.abs(np.degrees(theta)) <= ang + 1e-9))

        for k in np.nonzero(good)[0]:
            j = i0 + int(k)
            M = B_M[j]
            key = (tuple(N.flatten()), tuple(M.flatten()))
            if key in seen:
                continue
            seen.add(key)
            eps = np.array([[exx[k], exy[k]], [exy[k], eyy[k]]])
            matches.append(
                InterfaceMatch(
                    N=N.copy(),
                    M=M.copy(),
                    theta=float(theta[k]),
                    strain=eps,
                    area=areaV,
                    n_atoms_A=int(round(abs(np.linalg.det(N)))) * nat_A,
                    n_atoms_B=int(round(abs(np.linalg.det(M)))) * nat_B,
                )
            )

    # Sort by strain first (rounded, so numerically-equal strains don't let a
    # large cell jump ahead of a small one), then by area, then atom count.
    matches.sort(key=lambda m: (round(m.max_abs_strain, 6),
                                round(m.area, 6),
                                m.n_atoms))
    return matches


# ----------------------------------------------------------------------------
# Turn an accepted match into a real atomic interface (requires ASE)
# ----------------------------------------------------------------------------
def build_interface(
    surface_A,
    surface_B,
    match: InterfaceMatch,
    distance: float = 2.0,
    vacuum: float = 10.0,
    strain_B: bool = True,
    tag_slabs: bool = True,
):
    """
    Build the atomic interface for an accepted ``InterfaceMatch`` (eq 8).

    A* is built unchanged; B* is built, rotated to align with A*, then strained
    so its in-plane cell coincides with A*'s. The two slabs are stacked along z
    with ``distance`` Angstrom between them and ``vacuum`` Angstrom of vacuum.

    Parameters
    ----------
    surface_A, surface_B : ase.Atoms
        The same slabs passed to ``find_interfaces``.
    match : InterfaceMatch
    distance : float
        Gap (Angstrom) between the top of A and the bottom of B.
    vacuum : float
        Vacuum padding (Angstrom) added on each side along z.
    strain_B : bool
        Must match the value used in ``find_interfaces``.
    tag_slabs : bool
        If True (default), stamp a provenance marker so the two materials can be
        coloured separately in a viewer:
          * ``tags``  -> 1 for every atom of A, 2 for every atom of B
                         (ASE GUI / OVITO can colour directly "by tag");
          * array ``"slab"`` -> 0 for A, 1 for B (a clean boolean-ish handle);
          * array ``"orig_tags"`` -> the tags the input slabs already carried
                         (e.g. layer indices), preserved so nothing is lost.
        If False, the original ``tags`` are passed through unchanged.

    Returns
    -------
    ase.Atoms
        The interface supercell.
    """
    from ase.build import make_supercell

    def _P(mat2x2):
        P = np.eye(3, dtype=int)
        P[:2, :2] = mat2x2
        return P

    A_sc = make_supercell(surface_A, _P(match.N))
    B_sc = make_supercell(surface_B, _P(match.M))

    # The slab that gets deformed is the one that was strained in the search.
    fixed, moving = (A_sc, B_sc) if strain_B else (B_sc, A_sc)

    # Align the moving slab's first in-plane vector with the fixed one,
    # directly from the realised cells (robust to vector conventions).
    f1 = fixed.cell[0, :2]
    m1 = moving.cell[0, :2]
    theta = _angle(f1) - _angle(m1)
    moving.rotate(np.degrees(theta), "z", rotate_cell=True)

    # Apply the strain: overwrite the moving in-plane cell with the fixed one
    # and let ASE scale the atomic positions accordingly (this realises 1+eps).
    new_cell = moving.cell.copy()
    new_cell[:2, :2] = fixed.cell[:2, :2]
    moving.set_cell(new_cell, scale_atoms=True)

    # Provenance tagging so the two materials can be coloured separately.
    # Done after make_supercell (so counts are final) and before concatenation
    # (so ASE's array-union in `+` carries every array on both slabs).
    if tag_slabs:
        for slab, slab_id in ((A_sc, 0), (B_sc, 1)):
            n = len(slab)
            slab.set_array("orig_tags", slab.get_tags().copy())  # keep originals
            slab.set_array("slab", np.full(n, slab_id, dtype=int))
            slab.set_tags(np.full(n, slab_id + 1, dtype=int))    # 1 = A, 2 = B

    # Stack with A on the bottom and B on top (regardless of which was strained).
    bottom, top = A_sc, B_sc

    # Stack along z.
    shift = (bottom.positions[:, 2].max() - top.positions[:, 2].min()) + distance
    top.positions[:, 2] += shift

    interface = bottom + top

    # Common in-plane cell (they now match) + z height with vacuum.
    cell = bottom.cell.copy()
    zspan = interface.positions[:, 2].max() - interface.positions[:, 2].min()
    cell[2] = [0.0, 0.0, zspan + 2.0 * vacuum]
    interface.set_cell(cell)
    interface.center(axis=2)
    interface.pbc = (True, True, True)
    return interface


# ----------------------------------------------------------------------------
# Convenience: pretty-print a result table
# ----------------------------------------------------------------------------
def print_matches(matches: Sequence[InterfaceMatch], limit: int = 20) -> None:
    print(f"{'#':>3}  {'area/A^2':>9}  {'atoms':>5}  {'theta/deg':>9}  "
          f"{'eps_xx':>8}  {'eps_yy':>8}  {'eps_xy':>8}  {'max|eps|':>8}")
    for i, m in enumerate(matches[:limit]):
        e = m.strain
        print(f"{i:>3}  {m.area:9.2f}  {m.n_atoms:5d}  {m.theta_deg:9.2f}  "
              f"{e[0,0]:+8.4f}  {e[1,1]:+8.4f}  {e[0,1]:+8.4f}  "
              f"{m.max_abs_strain*100:7.2f}%")
    if len(matches) > limit:
        print(f"... ({len(matches) - limit} more)")


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Geometry-only demo (no ASE needed): graphene on a hexagonal metal ---
    def hexL(a):
        return np.array([[a, 0.0], [a / 2, a * np.sqrt(3) / 2]])

    graphene = hexL(2.46)     # surface A in-plane lattice (Angstrom)
    metal    = hexL(2.77)     # surface B in-plane lattice (Angstrom)

    print("Graphene / metal(111) interface search (geometry only)")
    matches = find_interfaces(graphene, metal,
                              max_strain=0.05,   # 5 % per strain component
                              max_area=400.0,    # cap interface cell area
                              max_angle=40.0)    # allow up to 40 deg twist
    print(f"\n{len(matches)} candidate interfaces found "
          f"(note how a small twist buys a much lower strain):\n")
    print_matches(matches, limit=15)

    # --- Full atomic build with ASE (uncomment in an environment with ASE) ---
    #
    # from ase.build import graphene as make_graphene, fcc111
    # from interface_matcher import find_interfaces, build_interface
    #
    # A = make_graphene(vacuum=0.0)            # slab, normal along z
    # B = fcc111("Ni", size=(1, 1, 3), a=3.52, vacuum=0.0)
    #
    # matches = find_interfaces(A, B, max_strain=0.03, max_area=120, max_angle=40)
    # iface = build_interface(A, B, matches[0], distance=2.1, vacuum=12.0)
    # from ase.io import write
    # write("interface.xyz", iface)
