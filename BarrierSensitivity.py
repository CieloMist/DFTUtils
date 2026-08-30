"""
BarrierSensitivity.py
=====================

Strain sensitivity of a solid-state transformation barrier from a MACE potential, by
autograd. Optional D3 dispersion (see `D3Correction`).

    import BarrierSensitivity as bs

What this computes
------------------
FIRST order -- `barrier_sensitivities()`:
    A_fwd = d(forward barrier)/d(strain)     kinetic
    A_rxn = d(reaction energy)/d(strain)     thermodynamic driving force
    A_rev = d(reverse barrier)/d(strain)     dependent on the other two
Each is a 3x3 tensor in eV per unit strain, obtained as a difference of virials.

SECOND order -- `activation_elastic_tensor()`:
    C_act = d2(barrier)/d(strain)2, a 3x3x3x3 tensor in eV per unit strain squared.

so the barrier expands as

    dE(eta) = dE(0) + A_fwd : eta + (1/2) eta : C_act : eta + ...

Why the first order needs no differentiation through the NEB
------------------------------------------------------------
All three geometries are stationary points, so by the envelope theorem
dE*/deta = partial E/partial eta with the geometry held fixed. Each configuration
contributes exactly one tensor -- its virial W = dE/deta = V*sigma -- and every first-order
output is a difference of virials. Atoms DO relax under strain; that relaxation simply
contributes nothing to first order, because it moves along directions in which the energy
is stationary.

This holds at ANY strain, not just infinitesimal -- but A is itself a function of eta, so
to get the sensitivity at finite strain you must re-relax ALL THREE configurations at that
strain (endpoints by minimisation, saddle by dimer) and recompute there.

The second order is different: it genuinely needs the geometry response,

    d2F/deta2 = E_eta,eta - E_eta,u H^-1 E_u,eta

the clamped-ion / relaxed-ion decomposition. See section 7b.

How strain reaches the graph
----------------------------
MACE's `displacement` hook: `prepare_graph()` reads `data.get("displacement")` and, when
compute_stress=True, applies

    positions <- positions + positions @ sym(eta)
    cell      <- cell      + cell      @ sym(eta)
    shifts    <- unit_shifts @ cell_strained

so `eta` is a genuine autograd leaf. `D3Correction` reproduces the same three lines against
torch-dftd's differentiable core.

THE VIRIAL SUMS OVER EVERY TERM IN THE TOTAL ENERGY, exactly like the forces do. If the PES
is MACE + D3 then W, the stationarity gate, and the eigenvalue gate all need their D3
partners. Pass a `D3Correction`; omit it and everything is MACE-only.

Validity
--------
Converged stationary points, a fixed mechanism, and a fixed neighbour list. The eigenvalue
gate (0, 1, 0) and the free-atom fmax gate check the first two. Nothing here detects
mechanism crossover, which is what actually bounds the strain range.

Requires: mace-torch, ase, torch. (+ torch-dftd for D3.) CPU is fine.
"""

from __future__ import annotations
 
import numpy as np
import torch
 
# ---------------------------------------------------------------------------------------
# CRITICAL: set the default dtype BEFORE constructing any AtomicData.
# AtomicData.from_config() calls torch.get_default_dtype() internally. If you set this after
# building the batch you get float32 tensors fed to a float64 model, and the resulting
# gradients are garbage at the 1e-4 level -- which is exactly the size of the effect you are
# trying to measure.
#
# Note mace_mp() defaults to default_dtype="float32". Pass default_dtype="float64" when you
# build the calculator, or your band is only converged to ~1e-3 eV/A and that residual force
# IS the envelope-theorem error term.
# ---------------------------------------------------------------------------------------
torch.set_default_dtype(torch.float64)
 
from mace.data import AtomicData, config_from_atoms  # noqa: E402
from mace.tools import AtomicNumberTable  # noqa: E402
from mace.tools.torch_geometric.batch import Batch  # noqa: E402
 
 
# =======================================================================================
# 1. Model loading and unwrapping
# =======================================================================================
def _iter_subcalcs(obj):
    """Yield child calculators of an ASE mixing calculator, across ASE versions.
 
    ASE >= 3.23: LinearCombinationCalculator holds a single `Mixer` OBJECT in `.mixer`,
                 and the children live at `.mixer.calcs`.
    older ASE  : the children sit directly on the calculator as `.calcs`.
 
    Note `.mixer` is not itself iterable -- iterating it raises
    "TypeError: 'Mixer' object is not iterable".
    """
    mixer = getattr(obj, "mixer", None)
    if mixer is not None:
        for sub in getattr(mixer, "calcs", None) or []:
            yield sub
    for attr in ("calcs", "calculators"):
        val = getattr(obj, attr, None)
        if isinstance(val, (list, tuple)):
            for sub in val:
                yield sub
 
def describe_calculator(obj, _depth=0):
    """Print the calculator tree. Use this if the extractors below fail on your ASE."""
    pad = "  " * _depth
    extra = " [has .models]" if hasattr(obj, "models") else ""
    print(f"{pad}{type(obj).__name__}{extra}")
    for sub in _iter_subcalcs(obj):
        describe_calculator(sub, _depth + 1)

def extract_mace_module(obj):
    """Pull the raw MACE torch module out of whatever mace_mp() handed back.
 
    mace_mp(dispersion=True) returns a SumCalculator([MACECalculator, TorchDFTD3Calculator]),
    not a MACECalculator, so `calc.models` does not exist at the top level.
    """
    if isinstance(obj, torch.nn.Module):
        return obj
    if hasattr(obj, "models"):                          # MACECalculator
        return obj.models[0]
    for sub in _iter_subcalcs(obj):                     # SumCalculator variants
        try:
            return extract_mace_module(sub)
        except TypeError:
            continue
    raise TypeError(f"no MACE module found inside {type(obj).__name__}")
 
 
def load_mace(model_or_path, device="cpu", dtype=torch.float64, freeze=True):
    """Load a raw MACE module (NOT MACECalculator, NOT MaceTorchSimModel).
 
    Accepts a path, a torch.nn.Module, or any calculator `extract_mace_module` understands.
 
    Do not enable cuEquivariance / OpenEquivariance / torch.compile on this instance. Those
    fused kernels are opaque to AOTAutograd and are not guaranteed to support the
    double-backward you need for Hessians and mixed second derivatives.
    """
    if isinstance(model_or_path, (str, bytes)) or hasattr(model_or_path, "__fspath__"):
        model = torch.load(model_or_path, map_location=device, weights_only=False)
    else:
        model = extract_mace_module(model_or_path)
 
    # NOTE .to() is in-place on module parameters. If you passed in calc.models[0], this
    # changes the calculator's dtype too. Do it after the NEB, not during.
    model = model.to(device=device, dtype=dtype).eval()
    if freeze:
        # freeze=False if you ever want dE/d(weights) -- barrier sensitivity to the model
        # parameters, e.g. for committee uncertainty propagation.
        for p in model.parameters():
            p.requires_grad_(False)
    return model
 
 
# =======================================================================================
# 2. Optional D3 dispersion term
# =======================================================================================
 
class D3Correction:
    """The D3 contribution to the energy, forces, virial, and Hessian -- all by autograd.

    D3 responds to affine deformation: straining the cell moves the atoms, which changes
    every interatomic distance D3 depends on, so dE_D3/deta is nonzero and D3 carries a
    genuine virial.

    HOW THIS REACHES THE GRAPH. torch-dftd's ASE calculator deliberately blocks autograd in
    two places: `calc_energy()` wraps the call in `torch.no_grad()`, and
    `calc_energy_and_forces()` runs its own `autograd.grad` and returns detached arrays.
    The differentiable core underneath both is `dftd_module.calc_energy_batch(...)`, which
    is pure torch and returns eV. We call it directly.

    The strain enters exactly the way MACE's `prepare_graph` does it:

        positions <- positions + positions @ sym(eta)
        cell      <- cell      + cell      @ sym(eta)
        shift_pos <- S @ cell_strained

    That last line is the one that matters. `_preprocess_atoms` builds it as
    `torch.mm(S, cell.detach())` -- the .detach() is precisely what severs the cell
    gradient. Recomputing it from the strained cell is what makes eta a real leaf.

    The edge list is built once, in numpy, and held FIXED. Same assumption as any analytic
    stress and as the MACE neighbour list: exact for a fixed edge set, and only wrong if a
    finite strain step pushes a pair across the cutoff, which the infinitesimal derivative
    never does.

    First derivatives agree with the old V*sigma path to ~1e-14, so this is a strict
    upgrade rather than a change of physics. What it adds is SECOND derivatives: the D3
    Hessian without 6N finite differences, and d2E/deta2 for the activation elastic tensor.

    All evaluations strip ASE constraints first -- the stationarity gate needs raw forces,
    and ASE zeroes constrained components by default.
    """

    def __init__(self, calc):
        self.calc = calc
        self.module = calc.dftd_module
        self.damping = calc.damping
        self.cutoff = calc.cutoff
        self.bidirectional = calc.bidirectional
        self.device = calc.device
        self.dtype = calc.dtype

    @classmethod
    def from_calculator(cls, obj):
        """Extract the D3 calculator from a SumCalculator, e.g. mace_mp(dispersion=True)."""
        if "d3" in type(obj).__name__.lower():
            return cls(obj)
        for sub in _iter_subcalcs(obj):
            try:
                return cls.from_calculator(sub)
            except TypeError:
                continue
        raise TypeError(
            f"no D3 calculator found inside {type(obj).__name__}. "
            f"Did you pass dispersion=True to mace_mp()?"
        )

    def graph(self, atoms, requires_grad=False):
        """Build the fixed edge list once. Returns tensors ready for calc_energy_batch.

        Constraints are stripped here: the edge list and every derivative below are raw.
        """
        from torch_dftd.functions.edge_extraction import calc_edge_index

        a = atoms.copy()
        a.set_constraint()

        pos = torch.tensor(a.get_positions(), dtype=self.dtype, device=self.device,
                           requires_grad=requires_grad)
        cell = torch.tensor(np.asarray(a.get_cell()), dtype=self.dtype, device=self.device)
        Z = torch.tensor(a.get_atomic_numbers(), device=self.device)
        pbc = torch.tensor(a.pbc, device=self.device)

        # Edge extraction must see plain positions, not a graph node.
        edge_index, S = calc_edge_index(
            pos.detach(), cell, pbc, cutoff=self.cutoff, bidirectional=self.bidirectional
        )
        return dict(pos=pos, cell=cell, Z=Z, pbc=pbc,
                    edge_index=edge_index, S=S.to(self.dtype))

    def energy_tensor(self, graph, strain=None):
        """Differentiable D3 energy, scalar tensor, eV.

        strain : optional [3, 3] leaf. Applied as the symmetric affine deformation above.
        """
        pos, cell = graph["pos"], graph["cell"]
        if strain is not None:
            sym = 0.5 * (strain + strain.transpose(-1, -2))
            pos = pos + pos @ sym
            cell = cell + cell @ sym
        shift_pos = graph["S"] @ cell        # NOT cell.detach() -- see class docstring
        E = self.module.calc_energy_batch(
            graph["Z"], pos, graph["edge_index"], cell, graph["pbc"], shift_pos,
            damping=self.damping,
        )
        return E.sum()

    def energy_from_leaves(self, atoms, pos, cell, strain=None):
        """D3 energy expressed in terms of leaves someone ELSE owns.

        Needed for the activation elastic tensor: the mixed derivative d2E/dR deta only
        makes sense if MACE and D3 differentiate the SAME position and strain tensors.
        `energy_tensor` builds its own leaves, which would give two disconnected graphs
        whose second derivatives cannot be combined.

        pos  : [n_atoms, 3] leaf, the REFERENCE (unstrained) positions
        cell : [3, 3] the reference cell
        strain : optional [3, 3] leaf
        """
        from torch_dftd.functions.edge_extraction import calc_edge_index

        a = atoms.copy()
        a.set_constraint()
        Z = torch.tensor(a.get_atomic_numbers(), device=self.device)
        pbc = torch.tensor(a.pbc, device=self.device)

        # Edge set is fixed and built off the graph, exactly as in `graph()`.
        edge_index, S = calc_edge_index(
            pos.detach(), cell.detach(), pbc,
            cutoff=self.cutoff, bidirectional=self.bidirectional,
        )
        S = S.to(self.dtype)

        if strain is not None:
            sym = 0.5 * (strain + strain.transpose(-1, -2))
            pos = pos + pos @ sym
            cell = cell + cell @ sym
        shift_pos = S @ cell

        E = self.module.calc_energy_batch(
            Z, pos, edge_index, cell, pbc, shift_pos, damping=self.damping,
        )
        return E.sum()

    def energy(self, atoms) -> float:
        with torch.no_grad():
            return float(self.energy_tensor(self.graph(atoms)))

    def forces(self, atoms) -> np.ndarray:
        g = self.graph(atoms, requires_grad=True)
        E = self.energy_tensor(g)
        (dE,) = torch.autograd.grad(E, g["pos"])
        return -dE.detach().cpu().numpy()

    def virial(self, atoms) -> np.ndarray:
        """dE_D3/deta. Same sign convention as the MACE virial, so directly additive.

        Equals V * sigma_D3 from the ASE stress path; fd_check_d3_virial() remains a
        useful independent check, but it is no longer the mechanism.
        """
        g = self.graph(atoms)
        strain = torch.zeros(3, 3, dtype=self.dtype, device=self.device,
                             requires_grad=True)
        E = self.energy_tensor(g, strain=strain)
        (W,) = torch.autograd.grad(E, strain)
        return W.detach().cpu().numpy()

    def hessian(self, atoms, chunk=1) -> np.ndarray:
        """d2E_D3/dR2 by autograd. Exact, and with no step size to choose.

        This replaces 6N finite-difference evaluations, but do NOT assume it is faster --
        that has not been measured on this system. Each of the 3N backward passes has to
        traverse the full double-backward graph over the whole D3 edge list (~460 edges per
        atom at the 21 A cutoff), which is not obviously cheaper than a fresh D3 call.
        Benchmark both on a compute node before committing to either.

        chunk : rows per backward.
                1  -> plain loop. Slowest per row, but lowest memory and always works.
                >1 -> vmap the backward over `chunk` rows at once via is_grads_batched.
                      Potentially much faster, but memory scales with `chunk`, and vmap
                      does not support every op -- if torch-dftd trips it, this falls back
                      to the loop automatically and says so.
        """
        g = self.graph(atoms, requires_grad=True)
        E = self.energy_tensor(g)
        (grad,) = torch.autograd.grad(E, g["pos"], create_graph=True)
        grad = grad.reshape(-1)
        n3 = grad.numel()

        H = torch.zeros(n3, n3, dtype=self.dtype, device=self.device)
        if chunk > 1:
            try:
                for start in range(0, n3, chunk):
                    rows = list(range(start, min(start + chunk, n3)))
                    basis = torch.zeros(len(rows), n3, dtype=self.dtype,
                                        device=self.device)
                    basis[range(len(rows)), rows] = 1.0
                    (out,) = torch.autograd.grad(
                        grad, g["pos"], grad_outputs=basis,
                        is_grads_batched=True, retain_graph=True,
                    )
                    H[rows] = out.reshape(len(rows), -1)
                H = H.detach().cpu().numpy()
                return 0.5 * (H + H.T)
            except (RuntimeError, NotImplementedError) as exc:
                print(f"  batched D3 Hessian unavailable ({type(exc).__name__}), "
                      f"falling back to the row loop")

        for k in range(n3):
            (row,) = torch.autograd.grad(grad[k], g["pos"], retain_graph=True)
            H[k] = row.reshape(-1)
        H = H.detach().cpu().numpy()
        return 0.5 * (H + H.T)

    def strain_hessian(self, atoms) -> np.ndarray:
        """d2E_D3/deta2, shape (3, 3, 3, 3) -- the D3 part of the elastic tensor.

        Only reachable because D3 now sits in the graph. This is the piece the activation
        elastic tensor needs and that reading sigma could never give.
        """
        g = self.graph(atoms)
        strain = torch.zeros(3, 3, dtype=self.dtype, device=self.device,
                             requires_grad=True)
        E = self.energy_tensor(g, strain=strain)
        (W,) = torch.autograd.grad(E, strain, create_graph=True)
        W = W.reshape(-1)

        C = torch.zeros(9, 9, dtype=self.dtype, device=self.device)
        for k in range(9):
            (row,) = torch.autograd.grad(W[k], strain, retain_graph=True)
            C[k] = row.reshape(-1)
        return C.detach().cpu().numpy().reshape(3, 3, 3, 3)
 
 
# =======================================================================================
# 3. ASE -> MACE translation
# =======================================================================================
 
class MaceBatcher:
    """Turns a list of ASE Atoms into a single batched MACE input dict.
 
    Batching all images into ONE forward pass is the main performance win over looping
    the ASE calculator per image.
    """
 
    def __init__(self, model, head: str | None = None):
        self.z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
        self.r_max = float(model.r_max)
        self.heads = list(getattr(model, "heads", ["Default"]))
        # A multi-head model is a different PES per head. Defaulting to heads[0] silently
        # evaluates a DIFFERENT functional than the NEB used, and NOTHING downstream catches
        # it: W is still symmetric, W still equals V*sigma, the atomic virial sum rule still
        # holds, A_fwd - A_rev still equals A_rxn. All of those are internal to MACE. The
        # only symptom is an inflated fmax, which you would misread as under-convergence.
        # So: if the model has more than one head, make the caller name it.
        if head is None and len(self.heads) > 1:
            raise ValueError(
                f"model exposes {len(self.heads)} heads {self.heads}; pass head=... "
                f"explicitly (use the SAME head the NEB was converged with)"
            )
        self.head = head if head is not None else self.heads[0]
        if self.head not in self.heads:
            raise ValueError(f"head {self.head!r} not in model heads {self.heads}")
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
 
    def build(self, atoms_list) -> dict:
        """Returns a plain dict ready for model(...).
 
        The neighbour list is built here, in numpy, from the current positions and is not
        differentiable. That is correct and is the same assumption behind any analytic
        stress: the derivative is exact for a fixed edge set.
        """
        datas = []
        for atoms in atoms_list:
            cfg = config_from_atoms(atoms, head_name=self.head)
            datas.append(
                AtomicData.from_config(
                    cfg, z_table=self.z_table, cutoff=self.r_max, heads=self.heads
                )
            )
        batch = Batch.from_data_list(datas).to(self.device)
        return batch.to_dict()
 
 
# =======================================================================================
# 4. The autograd hooks
# =======================================================================================
 
def evaluate(model, batch_dict, strain=None, atomic=False):
    """One batched forward pass with the graph kept alive.
 
    training=True is MANDATORY, and not for the reason you might assume. MACE computes
    forces/virials internally via torch.autograd.grad(..., retain_graph=training). With
    training=False the graph from energy back to positions and displacement is FREED by
    MACE's own call, and any subsequent grad() of yours raises
    "Trying to backward through the graph a second time".
    """
    # prepare_graph() WRITES BACK into the dict you hand it
    #     data["positions"], data["shifts"] = p, s
    # so reusing a dict double-applies the strain. Always pass a fresh shallow copy.
    d = dict(batch_dict)
    if strain is not None:
        d["displacement"] = strain
 
    return model(
        d,
        training=True,
        compute_force=True,
        compute_stress=True,          # required, or the displacement hook is ignored
        compute_edge_forces=atomic,   # needed to get atomic virials
        compute_atomic_stresses=atomic,
    )
 
 
def energies_and_virials(model, batch_dict, n_systems, atomic=False):
    """Returns (E [n], W [n,3,3], out, strain) -- MACE contribution only.
 
    W[x] = dE_x/deta, the virial of configuration x.
    """
    strain = torch.zeros(
        n_systems, 3, 3,
        dtype=batch_dict["positions"].dtype,
        device=batch_dict["positions"].device,
        requires_grad=True,
    )
    out = evaluate(model, batch_dict, strain=strain, atomic=atomic)
    E = out["energy"]                                   # [n_systems]
 
    # A single backward on E.sum() yields all n virials with no cross-contamination,
    # because each E_x depends only on its own 3x3 slice of `strain`.
    (W,) = torch.autograd.grad(E.sum(), strain, retain_graph=True)
    return E.detach(), W.detach(), out, strain
 
 
# =======================================================================================
# 5. Main entry point
# =======================================================================================
 
def barrier_sensitivities(
    model, batcher, atoms_initial, atoms_saddle, atoms_final,
    d3: D3Correction | None = None,
    atomic=True, fmax_tol=2e-3, verbose=True,
):
    """Three converged geometries in, two (plus one dependent) sensitivity tensors out.
 
    d3 : optional D3Correction. If your NEB was converged on MACE+D3 you MUST pass it, or
         (a) the reported energies and barriers omit dispersion, (b) W omits the D3 virial,
         and (c) the stationarity gate measures the wrong residual force and will warn for
         a reason you would misdiagnose.
 
    Returns a dict. All gradients are in eV per UNIT strain -- divide by 100 for
    eV per 1% strain, which is the number worth quoting.
    """
    images = (atoms_initial, atoms_saddle, atoms_final)
    labels = ("initial", "saddle", "final")
    _assert_consistent(*images)
 
    batch = batcher.build(list(images))
    E_mace, W_mace, out, _ = energies_and_virials(model, batch, n_systems=3, atomic=atomic)
    ptr = batch["ptr"]
    dt, dev = W_mace.dtype, W_mace.device
 
    # ---- MACE-only validation (must hold regardless of D3) -----------------------------
    # (a) The hook symmetrises, so every W must be symmetric. If not, the strain never
    #     reached the model and you are differentiating a zero tensor.
    for k, lbl in enumerate(labels):
        assert torch.allclose(W_mace[k], W_mace[k].T, atol=1e-9), \
            f"W[{lbl}] not symmetric -- displacement hook did not take effect"
    assert W_mace.abs().max() > 0, "all virials are zero -- check compute_stress=True"
 
    # (b) W must equal V*sigma. Independent code path inside MACE, so a genuine cross-check.
    #     Holds for slabs too: MACE defines stress = virial/V, so V*sigma reconstructs the
    #     virial exactly and vacuum padding cancels. Only sigma ALONE is vacuum-diluted.
    V = torch.linalg.det(batch["cell"].view(-1, 3, 3)).abs()
    W_from_stress = V.view(-1, 1, 1) * out["stress"].detach()
    assert torch.allclose(W_mace, W_from_stress, atol=1e-8), \
        f"W != V*sigma (max dev {(W_mace - W_from_stress).abs().max():.2e})"
 
    # ---- add the D3 term to every quantity it touches ----------------------------------
    F = out["forces"].detach()
    if d3 is not None:
        E_d3 = torch.tensor([d3.energy(a) for a in images], dtype=dt, device=dev)
        W_d3 = torch.stack(
            [torch.as_tensor(d3.virial(a), dtype=dt, device=dev) for a in images]
        )
        F_d3 = [torch.as_tensor(d3.forces(a), dtype=dt, device=dev) for a in images]
        E, W = E_mace + E_d3, W_mace + W_d3
    else:
        E_d3 = W_d3 = None
        F_d3 = [None] * 3
        E, W = E_mace, W_mace
 
    # ---- stationarity gate -------------------------------------------------------------
    # The envelope theorem needs dE_total/dR = 0 at all three points. The dropped term
    # scales as |F_resid| * ||dR*/deta||. Recompute fmax HERE rather than trusting the NEB:
    # if you converged with a different neighbour-list backend the residual on this PES may
    # differ. These are RAW forces -- ASE zeroes constrained components.
    #
    # WHICH forces to gate on, when some atoms are held by FixAtoms:
    #
    #   Split each atom's motion under strain into affine + internal relaxation u. The
    #   displacement hook already applies the affine part to EVERY atom, fixed ones included,
    #   so that contribution is inside W. What is left is
    #       dE/deta = W  +  (dE/dR_free) . (du_free/deta)
    #   and only the FREE-atom gradient multiplies a nonzero du. A FixAtoms component is a
    #   constraint force, not a residual error: that coordinate is not a variable, and its
    #   affine response is already accounted for. Gating on RAW force here would condemn a
    #   perfectly converged slab, since the frozen bulk region always carries large force.
    #
    #   This assumes the fixed region deforms AFFINELY with the cell -- correct for a
    #   strain-controlled slab anchored to bulk. If you instead mean to pin those atoms in
    #   absolute Cartesian space under strain, the hook's affine motion is not what you want
    #   and W is the wrong tensor.
    #
    #   Raw fmax is still computed and reported, just not gated on.
    fmax, fmax_raw = [], []
    for k in range(3):
        f = F[ptr[k]:ptr[k + 1]]
        if F_d3[k] is not None:
            f = f + F_d3[k]
        fn = f.norm(dim=1)
        fmax_raw.append(fn.max().item())
        m = torch.as_tensor(_free_mask(images[k]), device=fn.device)
        fmax.append(fn[m].max().item() if bool(m.any()) else 0.0)
    n_free = int(_free_mask(images[0]).sum())
    if n_free < len(images[0]):
        print(f"  {n_free} of {len(images[0])} atoms free; gating on free-atom fmax "
              f"(raw includes FixAtoms constraint forces and is not an error measure)")
    for lbl, fm in zip(labels, fmax):
        if fm > fmax_tol:
            print(f"  WARNING: {lbl} free-atom fmax = {fm:.2e} eV/A > {fmax_tol:.0e}. "
                  f"Envelope theorem error ~ this magnitude. Converge harder."
                  + ("" if d3 is not None else "  (D3 not included -- if your band was "
                                               "converged with dispersion, pass d3=)"))
 
    # ---- the outputs -------------------------------------------------------------------
    A_fwd = W[1] - W[0]     # forward barrier   E_s - E_i
    A_rxn = W[2] - W[0]     # reaction energy   E_f - E_i
    A_rev = W[1] - W[2]     # reverse barrier   E_s - E_f
 
    # Algebraically vacuous (all three come from the same W), but catches index and sign
    # typos for free. This is NOT validation -- see the tier-3 FD check.
    assert torch.allclose(A_fwd - A_rev, A_rxn, atol=1e-12)
 
    result = dict(
        includes_d3=d3 is not None,
        E_initial=E[0].item(), E_saddle=E[1].item(), E_final=E[2].item(),
        barrier_fwd=(E[1] - E[0]).item(),
        barrier_rev=(E[1] - E[2]).item(),
        E_rxn=(E[2] - E[0]).item(),        # 0 K internal energy. NOT a free energy.
        W_initial=W[0], W_saddle=W[1], W_final=W[2],
        W_mace=W_mace, W_d3=W_d3,          # kept separate for transparency
        A_fwd=A_fwd, A_rxn=A_rxn, A_rev=A_rev,
        A_fwd_voigt=to_voigt(A_fwd), A_rxn_voigt=to_voigt(A_rxn),
        fmax=dict(zip(labels, fmax)),                # free atoms -- the gated quantity
        fmax_raw=dict(zip(labels, fmax_raw)),        # all atoms, incl. constraint forces
        volume=V.detach(),
    )
 
    # ---- per-atom decomposition (MACE ONLY) --------------------------------------------
    if atomic:
        # MACE returns atomic_virials with the opposite sign to dE/deta (it follows the
        # `virials = -dE/deta` convention). Flip so it sums to +W_mace like everything else.
        w_atom = -out["atomic_virials"].detach()
        sums = torch.stack([w_atom[ptr[k]:ptr[k + 1]].sum(0) for k in range(3)])
        # Sum rule is against W_MACE, not W_total: D3 has its own pairwise partition but it
        # is not in MACE's atomic_virials, so this decomposition is honestly MACE-only.
        dev_max = (sums - W_mace).abs().max()
        assert dev_max < 1e-6, (
            f"atomic virial sum rule violated ({dev_max:.2e}). If it is off by exactly a "
            f"sign, drop the leading minus above -- MACE's virial sign conventions differ "
            f"between `virials` and the raw gradient."
        )
        n = len(atoms_initial)
        result["dw_fwd"] = w_atom[ptr[1]:ptr[2]] - w_atom[ptr[0]:ptr[1]]
        result["dw_rxn"] = w_atom[ptr[2]:ptr[3]] - w_atom[ptr[0]:ptr[1]]
        result["dw_is_mace_only"] = True
        assert result["dw_fwd"].shape == (n, 3, 3)
        # Caveat: the atomic partition is gauge-dependent -- only the total is unique.
        # Differences between configurations of the same system are more robust than
        # absolute per-atom values. Use for spatial localisation, not absolute claims.
 
    if verbose:
        _report(result)
    return result
 
 
# =======================================================================================
# 6. Reporting helpers
# =======================================================================================
 
def to_voigt(A):
    """3x3 gradient -> Voigt 6, conjugate to ENGINEERING strain.
 
    PLAIN COPY, no factor of 2. The energy differential is a full double contraction, so
    each off-diagonal pair contributes 2*A_xy*d(eta_xy); the engineering definition
    gamma_xy = 2*eta_xy absorbs exactly that factor. A converts like a stress, because
    A = V*sigma. It is the STRAIN vector that gets the 2, not the gradient.
    """
    return torch.stack([A[0, 0], A[1, 1], A[2, 2], A[1, 2], A[0, 2], A[0, 1]])
 
 
def sensitivity_along(A, direction):
    """Sensitivity to one loading mode, in eV per unit strain. Convention-free.
 
    Examples:
        uniaxial along n :  torch.outer(n, n)
        shear on (n, m)  :  0.5 * (torch.outer(n, m) + torch.outer(m, n))
        hydrostatic      :  torch.eye(3)
    """
    d = 0.5 * (direction + direction.T)
    d = d / d.norm()
    return (A * d).sum()
 
 
def _report(r):
    tag = "MACE + D3" if r["includes_d3"] else "MACE only"
    print(f"\n  PES: {tag}")
    print(f"  E_initial {r['E_initial']:14.6f} eV")
    print(f"  E_saddle  {r['E_saddle']:14.6f} eV")
    print(f"  E_final   {r['E_final']:14.6f} eV")
    print(f"\n  barrier (fwd) {r['barrier_fwd']:10.4f} eV")
    print(f"  barrier (rev) {r['barrier_rev']:10.4f} eV")
    print(f"  E_rxn         {r['E_rxn']:10.4f} eV   [0 K internal energy, no entropy/ZPE]")
    names = ["xx", "yy", "zz", "yz", "xz", "xy"]
    print("\n  sensitivity, eV per 1% strain (Voigt, engineering shear):")
    print(f"  {'':>6}" + "".join(f"{nm:>10}" for nm in names))
    for key, lbl in (("A_fwd_voigt", "barrier"), ("A_rxn_voigt", "E_rxn")):
        vals = (r[key] / 100.0).tolist()
        print(f"  {lbl:>6}" + "".join(f"{v:10.4f}" for v in vals))
    if r["includes_d3"] and r["W_d3"] is not None:
        frac = (r["W_d3"][1] - r["W_d3"][0]).abs().max() / max(
            r["A_fwd"].abs().max().item(), 1e-30)
        print(f"\n  D3 share of |A_fwd|_max: {100 * frac:.1f}%")
    print(f"\n  fmax (free): " + "  ".join(f"{k}={v:.1e}" for k, v in r["fmax"].items()))
    print(f"  fmax (raw) : " + "  ".join(f"{k}={v:.1e}" for k, v in r["fmax_raw"].items()))
 
 
def _free_mask(atoms) -> np.ndarray:
    """Boolean mask of atoms that are free to relax, i.e. the actual variables.

    Only FixAtoms is decoded. Anything else (FixedPlane, FixBondLength, ...) partially
    constrains an atom, and treating it as fully free is the conservative choice: the gate
    stays pessimistic rather than silently passing.
    """
    mask = np.ones(len(atoms), dtype=bool)
    for c in atoms.constraints:
        idx = getattr(c, "index", None)
        if idx is not None and type(c).__name__ == "FixAtoms":
            mask[np.asarray(idx)] = False
    return mask


def _assert_consistent(*atoms_list):
    ref = atoms_list[0]
    for i, a in enumerate(atoms_list[1:], 1):
        assert len(a) == len(ref), f"image {i}: atom count differs"
        assert (a.get_atomic_numbers() == ref.get_atomic_numbers()).all(), \
            f"image {i}: atom ORDER differs -- per-atom decomposition would be meaningless"
        assert np.allclose(a.get_cell(), ref.get_cell(), atol=1e-8), \
            f"image {i}: cell differs -- this code assumes fixed-cell (strain-controlled) NEB"
        # The envelope theorem drops (dE/dR_free).(du_free/deta), so all three geometries
        # must be stationary in the SAME free subspace. If the endpoints were relaxed under
        # a tighter constraint set than the band uses, the atoms the band frees but the
        # relaxation froze sit at unrelaxed positions with real residual force -- and every
        # other check here still passes. Costs nothing to catch it.
        assert set(np.flatnonzero(~_free_mask(a))) == set(np.flatnonzero(~_free_mask(ref))), \
            (f"image {i}: FixAtoms set differs from image 0 "
             f"({(~_free_mask(a)).sum()} vs {(~_free_mask(ref)).sum()} fixed). All images "
             f"must be relaxed in the same free subspace or the envelope theorem fails.")
 
 
# =======================================================================================
# 7. Validation
# =======================================================================================
 
def _affine(atoms, D, eps):
    a = atoms.copy()
    a.set_constraint()
    a.set_cell(np.asarray(atoms.get_cell()) @ (np.eye(3) + eps * D), scale_atoms=True)
    return a
 
 
def fd_check_virial(model, batcher, atoms, direction=None, h=1e-4):
    """Validate MACE's displacement hook end-to-end against a real deformation.
 
    Does NOT test the envelope theorem (for that you must re-converge NEBs at eta +/- h).
    DOES test that the hook, the symmetrisation, the cell/shift update and your sign
    conventions are all correct -- the part most likely to be wrong -- for the cost of
    three single-point energies.
    """
    if direction is None:
        direction = torch.tensor([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    D = 0.5 * (direction + direction.T)
 
    batch = batcher.build([atoms])
    _, W, _, _ = energies_and_virials(model, batch, n_systems=1)
    analytic = (W[0] * D).sum().item()
 
    Dn = D.numpy()
    def energy_at(eps):
        b = batcher.build([_affine(atoms, Dn, eps)])
        with torch.no_grad():
            return model(dict(b), training=False, compute_force=False,
                         compute_stress=False)["energy"][0].item()
 
    numeric = (energy_at(h) - energy_at(-h)) / (2 * h)
    rel = abs(analytic - numeric) / max(abs(numeric), 1e-12)
    print(f"  MACE virial FD check: analytic {analytic:.8f}  numeric {numeric:.8f}  "
          f"rel err {rel:.2e}")
    assert rel < 1e-5, "displacement hook does not match a real affine deformation"
    return analytic, numeric
 
 
def fd_check_d3_virial(d3: D3Correction, atoms, direction=None, h=1e-4):
    """Validate that V*sigma_D3 really is dE_D3/deta, in MACE's sign convention.
 
    This is the check that catches a sign or factor error in adding D3 to W. Run it once.
    """
    if direction is None:
        direction = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    D = 0.5 * (direction + direction.T)
 
    analytic = float((d3.virial(atoms) * D).sum())
    numeric = (d3.energy(_affine(atoms, D, h)) - d3.energy(_affine(atoms, D, -h))) / (2 * h)
    rel = abs(analytic - numeric) / max(abs(numeric), 1e-12)
    print(f"  D3 virial FD check:   analytic {analytic:.8f}  numeric {numeric:.8f}  "
          f"rel err {rel:.2e}")
    assert rel < 1e-4, (
        "V*sigma_D3 != dE_D3/deta. Check the ASE stress sign/units for your torch-dftd "
        "version before adding it to W."
    )
    return analytic, numeric
 
 
def positional_hessian(model, batcher, atoms, d3: D3Correction | None = None,
                       free_only=True):
    """d2E/dR2 for MACE (+ D3), optionally restricted to the free subspace.

    Hessians of separate energy terms simply add, so MACE's `compute_hessian` and
    D3Correction.hessian can be summed even though they come from different graphs.
    See CONSTRAINED_SUBSPACE.md for why the free-subspace restriction is exact for
    FixAtoms (an affine constraint) and would need correction terms for anything else.
    """
    batch = batcher.build([atoms])
    out = model(dict(batch), training=True, compute_force=True, compute_stress=False,
                compute_hessian=True)
    n = len(atoms)
    H = out["hessian"].reshape(3 * n, 3 * n).detach()
    H = 0.5 * (H + H.T)
    if d3 is not None:
        H = H + torch.as_tensor(d3.hessian(atoms), dtype=H.dtype, device=H.device)

    if free_only:
        dof = np.repeat(_free_mask(atoms), 3)          # (3N,) -> x,y,z per atom
        idx = torch.as_tensor(np.flatnonzero(dof), device=H.device)
        H = H.index_select(0, idx).index_select(1, idx)
    return H


def n_negative_eigenvalues(model, batcher, atoms, d3: D3Correction | None = None,
                           tol=1e-4, free_only=True):
    """Hessian eigenvalue count. MUST be 1 at the saddle and 0 at the endpoints.

    Cheapest reliable correctness alarm you have. Cost is O(3N) backward passes for MACE,
    plus 3N more for D3 if d3 is given.

    THE HESSIAN MUST BE RESTRICTED TO THE FREE SUBSPACE. MACE and torch-dftd both return the
    raw 3N x 3N Hessian -- neither knows anything about ASE constraints. But a FixAtoms atom
    is not a variable, and it is not at a stationary point either: it sits wherever it was
    pinned, carrying a large constraint force. Curvature along those directions is
    meaningless, and diagonalising the full matrix reports their negative eigenvalues as if
    they were unstable modes. On a 92-atom slab with 60 atoms fixed that showed up as "28
    negative eigenvalue(s)" at a minimum whose free-atom fmax was 4e-4.

    Same distinction as the fmax gate: what counts is the subspace the geometry was actually
    relaxed in. Restricting to H[free, free] is the constrained Hessian, and its signature is
    what the (0, 1, 0) rule is about.

    Note there are no translational zero modes in the free subspace when anything is fixed --
    the frozen region pins the system -- so `tol` only has to sit below real mode curvatures.
    Set free_only=False to inspect the raw matrix.
    """
    H = positional_hessian(model, batcher, atoms, d3=d3, free_only=free_only)
    evals = torch.linalg.eigvalsh(H)
    return int((evals < -tol).sum().item()), evals[:8]
 
 
# =======================================================================================
# 7b. Activation elastic tensor  C_act = d2(barrier)/deta2
# =======================================================================================
#
# The envelope theorem makes the FIRST derivative free at any strain: dF/deta = E_eta,
# evaluated at the geometry re-relaxed for that strain. It does NOT extend to the second
# derivative. Differentiating the envelope result again,
#
#     d2F/deta2 = E_eta,eta + E_eta,u (du*/deta)
#
# and du*/deta no longer multiplies something that vanishes. Implicit differentiation of
# the stationarity condition E_u = 0 gives du*/deta = -H^-1 E_u,eta, hence
#
#     d2F/deta2 = E_eta,eta  -  E_eta,u H^-1 E_u,eta
#                 ^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^
#                clamped-ion      internal relaxation
#
# the standard clamped-ion / relaxed-ion decomposition. C_act is the difference of this
# between the saddle and the initial state.
#
# SIGN STRUCTURE. In the eigenbasis of H the correction is sum_k (v_k . E_u,eta)^2 / lambda_k.
# At a minimum every lambda_k > 0, so subtracting it always SOFTENS. At the saddle the
# unstable mode has lambda_1 < 0, so that one term flips sign and STIFFENS. C_act is
# therefore not a difference of two softenings.
#
# RADIUS OF CONVERGENCE. As strain approaches a saddle-node bifurcation (saddle and minimum
# merging, the athermal limit) lambda_1 -> 0 and the 1/lambda_k term diverges. A first-order
# calculation stays well behaved right up to the fold and gives no warning; this one does.
# `cond` in the returned dict is the alarm.


def _clamped_and_mixed(model, batcher, atoms, d3: D3Correction | None = None):
    """Returns (E_ee [3,3,3,3], E_ue [3N,3,3]) for MACE + D3 on ONE shared graph.

    E_ee = d2E/deta2 at FIXED geometry (clamped-ion)
    E_ue = d2E/dR deta (the strain-force coupling)

    Both come from the same 9 backward passes, because differentiating dE/deta with respect
    to [strain, positions] yields both blocks at once.

    D3 must be added through `energy_from_leaves` rather than its own graph: a mixed second
    derivative is only meaningful if both terms differentiate the SAME leaves.
    """
    batch = batcher.build([atoms])
    pos = batch["positions"]
    pos.requires_grad_(True)
    cell = batch["cell"].reshape(3, 3)
    dt, dev = pos.dtype, pos.device

    strain = torch.zeros(1, 3, 3, dtype=dt, device=dev, requires_grad=True)
    out = evaluate(model, batch, strain=strain)
    E = out["energy"].sum()
    if d3 is not None:
        E = E + d3.energy_from_leaves(atoms, pos, cell, strain=strain[0])

    (g_eta,) = torch.autograd.grad(E, strain, create_graph=True)
    g_eta = g_eta.reshape(9)

    n3 = pos.numel()
    E_ee = torch.zeros(9, 9, dtype=dt, device=dev)
    E_ue = torch.zeros(n3, 9, dtype=dt, device=dev)
    for k in range(9):
        d_eta, d_u = torch.autograd.grad(g_eta[k], [strain, pos], retain_graph=True)
        E_ee[k] = d_eta.reshape(9)
        E_ue[:, k] = d_u.reshape(-1)

    E_ee = 0.5 * (E_ee + E_ee.T)          # symmetric by construction; enforce it
    return E_ee.reshape(3, 3, 3, 3), E_ue.reshape(n3, 3, 3)


def relaxed_elastic_tensor(model, batcher, atoms, d3: D3Correction | None = None):
    """Relaxed-ion d2E/deta2 for one configuration. Returns a dict."""
    E_ee, E_ue = _clamped_and_mixed(model, batcher, atoms, d3=d3)
    H = positional_hessian(model, batcher, atoms, d3=d3, free_only=True)

    dof = np.repeat(_free_mask(atoms), 3)
    idx = torch.as_tensor(np.flatnonzero(dof), device=H.device)
    B = E_ue.reshape(-1, 9).index_select(0, idx)        # [nfree3, 9]

    evals = torch.linalg.eigvalsh(H)
    cond = float(evals.abs().max() / evals.abs().min())

    X = torch.linalg.solve(H, B)                       # H^-1 E_u,eta
    correction = B.T @ X                               # E_eta,u H^-1 E_u,eta
    C = E_ee.reshape(9, 9) - correction

    return dict(
        C=C.reshape(3, 3, 3, 3).detach(),
        C_clamped=E_ee.reshape(3, 3, 3, 3).detach(),
        C_relaxation=correction.reshape(3, 3, 3, 3).detach(),
        n_negative=int((evals < -1e-4).sum().item()),
        lambda_min_abs=float(evals.abs().min()),
        cond=cond,
    )


def activation_elastic_tensor(model, batcher, atoms_initial, atoms_saddle,
                              d3: D3Correction | None = None, verbose=True):
    """C_act = C(saddle) - C(initial), the second-order barrier response to strain.

    Units are eV per unit strain SQUARED. The barrier expands as

        dE(eta) = dE(0)  +  A_fwd : eta  +  (1/2) eta : C_act : eta  +  ...

    Use `contract_C` to evaluate it along a loading mode.
    """
    r_i = relaxed_elastic_tensor(model, batcher, atoms_initial, d3=d3)
    r_s = relaxed_elastic_tensor(model, batcher, atoms_saddle, d3=d3)

    assert r_i["n_negative"] == 0, (
        f"initial state has {r_i['n_negative']} negative Hessian eigenvalues -- not a "
        f"minimum, so H^-1 does not mean what this formula assumes")
    assert r_s["n_negative"] == 1, (
        f"saddle has {r_s['n_negative']} negative Hessian eigenvalues, expected 1")

    C_act = r_s["C"] - r_i["C"]
    result = dict(
        C_act=C_act,
        C_act_clamped=r_s["C_clamped"] - r_i["C_clamped"],
        initial=r_i, saddle=r_s,
    )

    # Near a saddle-node bifurcation lambda_1 -> 0 at the saddle and the relaxation term
    # diverges. This is the only diagnostic in the pipeline that sees it coming.
    for lbl, r in (("initial", r_i), ("saddle", r_s)):
        if r["cond"] > 1e6:
            print(f"  WARNING: {lbl} Hessian condition number {r['cond']:.1e} "
                  f"(|lambda|_min = {r['lambda_min_abs']:.2e}). The relaxation term is "
                  f"near-singular -- approaching a bifurcation, and C is unreliable.")

    if verbose:
        names = ["xx", "yy", "zz", "yz", "xz", "xy"]
        print("\n  activation elastic tensor, eV per unit strain^2 (Voigt 6x6):")
        Cv = to_voigt_4(C_act).tolist()
        print(f"  {'':>5}" + "".join(f"{nm:>10}" for nm in names))
        for i, nm in enumerate(names):
            print(f"  {nm:>5}" + "".join(f"{v:10.1f}" for v in Cv[i]))
        # Split the LARGEST component of C_act into its two origins. Comparing
        # max|C_clamped| with max|C_act| would compare different components and mislead.
        flat = C_act.reshape(-1)
        k = int(flat.abs().argmax())
        tot = float(flat[k])
        cl = float(result["C_act_clamped"].reshape(-1)[k])
        print(f"\n  largest component {tot:+.1f} = clamped-ion {cl:+.1f} "
              f"({100 * cl / tot:.0f}%) + internal relaxation {tot - cl:+.1f} "
              f"({100 * (tot - cl) / tot:.0f}%)")
        print(f"  Hessian condition: initial {r_i['cond']:.1e}   saddle {r_s['cond']:.1e}")
    return result


def to_voigt_4(C):
    """[3,3,3,3] -> [6,6], conjugate to ENGINEERING strain. PLAIN COPY, no factors.

    Same rule as `to_voigt`, and for the same reason. With gamma_I = w_I * eta_(I), where
    w = (1,1,1,2,2,2) is the multiplicity of each index pair,

        (1/2) sum_ijkl eta_ij C_ijkl eta_kl = (1/2) sum_IJ w_I w_J eta_(I) C_(I)(J) eta_(J)
                                            = (1/2) sum_IJ gamma_I C_(I)(J) gamma_J

    so the multiplicity factors cancel against the engineering-strain factors exactly and
    C^V_IJ = C_(I)(J). Verified on pure shear: C^V_66 = C_1212.
    """
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    return torch.stack([
        torch.stack([C[i, j, k, l] for (k, l) in pairs]) for (i, j) in pairs
    ])


def contract_C(C, direction):
    """Second-order coefficient along one loading mode, eV per unit strain^2.

    Companion to `sensitivity_along`. The barrier along a normalised direction d is

        dE(s) = dE(0) + s * (A : d) + (1/2) s^2 * (d : C : d) + ...

    where s is the scalar strain amplitude.
    """
    d = 0.5 * (direction + direction.T)
    d = d / d.norm()
    return torch.einsum("ij,ijkl,kl->", d, C, d)

# =======================================================================================
# 8. Results IO -- write once here, read in a notebook
# =======================================================================================

def _jsonable(x):
    """torch tensors / numpy arrays -> nested lists; numpy scalars -> python scalars."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def write_results(results, destination="barrier_sensitivity_results.json",
                  elastic=None, extra=None):
    """Serialise everything to one JSON, ready to read straight into a notebook.

    results  : the dict from barrier_sensitivities()
    elastic  : optional dict from activation_elastic_tensor()
    extra    : optional dict of run metadata (head, strain state, paths, ...)

    JSON rather than pickle so the file stays inspectable, diffable, and readable without
    this module on the path. Tensors land as nested lists; `read_results` converts them
    back to numpy.
    """
    import json

    out = {k: _jsonable(v) for k, v in results.items()}
    out["order"] = 1

    # Voigt packings are the form you actually plot, so store them explicitly rather than
    # making the notebook rederive the index convention.
    names = ["xx", "yy", "zz", "yz", "xz", "xy"]
    out["voigt_labels"] = names
    for key in ("A_fwd", "A_rxn", "A_rev"):
        out[f"{key}_voigt"] = _jsonable(to_voigt(results[key]))

    if elastic is not None:
        out["order"] = 2
        out["C_act"] = _jsonable(elastic["C_act"])
        out["C_act_voigt"] = _jsonable(to_voigt_4(elastic["C_act"]))
        out["C_act_clamped_voigt"] = _jsonable(to_voigt_4(elastic["C_act_clamped"]))
        for lbl in ("initial", "saddle"):
            out[f"C_{lbl}_voigt"] = _jsonable(to_voigt_4(elastic[lbl]["C"]))
            out[f"C_{lbl}_clamped_voigt"] = _jsonable(to_voigt_4(elastic[lbl]["C_clamped"]))
            out[f"hessian_cond_{lbl}"] = elastic[lbl]["cond"]
            out[f"hessian_n_negative_{lbl}"] = elastic[lbl]["n_negative"]

    if extra:
        out.update(_jsonable(extra))

    with open(destination, "w") as f:
        json.dump(out, f, indent=2)
    return destination


def read_results(path="barrier_sensitivity_results.json"):
    """Inverse of write_results. Returns a dict with numpy arrays instead of nested lists.

    Notebook usage:

        import BarrierSensitivity as bs
        r = bs.read_results('Barrier_Sensitivity/barrier_sensitivity_results.json')
        plt.bar(r['voigt_labels'], r['A_fwd_voigt'] / 100)   # eV per 1% strain
    """
    import json

    with open(path) as f:
        r = json.load(f)
    for k, v in list(r.items()):
        if isinstance(v, list) and v and not isinstance(v[0], str):
            r[k] = np.array(v)
    return r
