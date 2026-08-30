# Handoff brief — MACE barrier strain-sensitivity project

Paste this into Claude Code as project context. It replaces a long design conversation.
Everything under "Verified API facts" was read from MACE/ASE/TorchSim source, not assumed.

Working file: `MACE_Barrier_Sensitivity.py` (remote server, VSCode).

---

## 1. Goal

Compute the sensitivity of a solid–solid phase-transformation energy barrier to unit-cell
strain, using a MACE machine-learned potential and PyTorch autodiff. Two outputs:

- `A_fwd` = d(forward barrier)/d(strain) — kinetic
- `A_rxn` = d(reaction energy)/d(strain) — thermodynamic driving force

Plus a dependent third, `A_rev` = d(reverse barrier)/d(strain). All are 3×3 symmetric
tensors (6 Voigt components), in eV per unit strain.

## 2. The key result — read this before proposing any architecture change

**Do NOT differentiate through the NEB.** It is unnecessary. At a converged stationary
point `∇E = 0`, so by the envelope theorem the implicit geometry-response term drops:

```
dΔE‡/dη  =  ∂E/∂η|_saddle  −  ∂E/∂η|_initial      (geometries held fixed)
```

Each configuration contributes exactly one tensor — its **virial** `W = ∂E/∂η = V·σ` — and
every output is a difference of virials. So the pipeline is:

1. Converge a climbing-image NEB in ASE (existing, separate machinery). Not differentiable,
   does not need to be.
2. Take three converged ASE `Atoms`: initial minimum, climbing image (saddle), final minimum.
3. One batched MACE forward + one backward → three virials → all outputs.

Consequences worth internalising:

- Unrolling the optimizer, `torch.utils.checkpoint`, implicit differentiation via GMRES —
  all unnecessary for **first** derivatives. They become necessary only for second
  derivatives (the activation elastic tensor `𝒞`) or path-dependent quantities.
- The NEB engine choice is a pure performance question. ASE is fine.
- Validity requires: converged stationary points, fixed mechanism, infinitesimal strain,
  fixed neighbour list. See §6.

## 3. Verified MACE API facts

**The strain hook.** `mace/modules/utils.py::prepare_graph` reads `data.get("displacement")`.
If you put a `[n_graphs, 3, 3]` leaf tensor there **and** pass `compute_stress=True`, MACE
applies `positions += positions @ sym(η)`, `cell += cell @ sym(η)`, and recomputes shifts —
making η a real autograd leaf. Undocumented upstream API; pin the MACE version.

**`training=True` is mandatory.** `compute_forces_virials` calls
`torch.autograd.grad(..., retain_graph=training, create_graph=training)`. With
`training=False` the graph is **freed by MACE's own call** and your subsequent
`grad(E, strain)` raises "Trying to backward through the graph a second time."

**`prepare_graph` mutates its input dict** (`data["positions"], data["shifts"] = p, s`).
Always pass a fresh `dict(batch_dict)` copy or the strain is applied twice.

**Sign conventions.**
- `out["stress"]` = `(∂E/∂η)/V`, so `W == V·σ` exactly. Use as a cross-check.
- `out["virials"]` = `−∂E/∂η` (opposite sign to `W`).
- `out["atomic_virials"]` follows the `virials` convention → negate to make it sum to `+W`.
- Atomic virials require **both** `compute_edge_forces=True` and `compute_atomic_stresses=True`.

**Batch construction** (matches what ASE's `MACECalculator` does, so same PES):
```python
cfg  = config_from_atoms(atoms, head_name=head)     # key_specification defaults to empty
data = AtomicData.from_config(cfg, z_table=z_table, cutoff=r_max, heads=heads)
d    = Batch.from_data_list([...]).to(device).to_dict()
```
`torch.set_default_dtype(torch.float64)` **before** any `AtomicData` — `from_config` reads
the global default dtype internally.

**Calculators.**
- `MACECalculator.models` is a **list** (committee support); params already frozen.
- `MACECalculator(models=[module])` accepts a pre-loaded module — use this to instantiate once.
- `mace_mp(dispersion=True)` returns `SumCalculator([MACECalculator, TorchDFTD3Calculator])`,
  **not** a `MACECalculator`. Source: `if not dispersion: return mace_calc`.
- `mace_mp` defaults to **`default_dtype="float32"`**. Pass `"float64"` or the band is only
  converged to ~1e-3 eV/Å, and that residual force *is* the envelope-theorem error term.
- ASE ≥3.23 `LinearCombinationCalculator` holds a `Mixer` **object** at `.mixer`; children are
  at `.mixer.calcs`. Older ASE puts `.calcs` on the calculator directly. `.mixer` is **not**
  iterable — iterating it raises `TypeError: 'Mixer' object is not iterable`.

**PolarMACE (future work).** `forward(..., external_field=None)` is an explicit kwarg;
`∂E/∂F = −μ`, so the field analogue of `W` is the dipole and the barrier output is the
activation dipole. Requires a polar checkpoint (MACE-MP is not polar) and the
`graph_electrostatics` package.

## 4. Current code state

Single module, ~8 sections:

| section | contents |
|---|---|
| 1 | `extract_mace_module`, `load_mace`, `_iter_subcalcs`, `describe_calculator` |
| 2 | `D3Correction` — optional dispersion term |
| 3 | `MaceBatcher` — ASE → MACE batch dict |
| 4 | `evaluate`, `energies_and_virials` — the autograd hooks |
| 5 | `barrier_sensitivities` — main entry point |
| 6 | `to_voigt`, `sensitivity_along`, `_report`, `_assert_consistent` |
| 7 | `fd_check_virial`, `fd_check_d3_virial`, `n_negative_eigenvalues` |
| 8 | `__main__` example |

**Latest patch (apply if not present)** — fixes `TypeError: 'Mixer' object is not iterable`:

```python
def _iter_subcalcs(obj):
    """Yield child calculators of an ASE mixing calculator, across ASE versions."""
    mixer = getattr(obj, "mixer", None)          # ASE >= 3.23: a Mixer OBJECT, not a list
    if mixer is not None:
        for sub in getattr(mixer, "calcs", None) or []:
            yield sub
    for attr in ("calcs", "calculators"):        # older ASE
        val = getattr(obj, attr, None)
        if isinstance(val, (list, tuple)):
            for sub in val:
                yield sub
```
`extract_mace_module` and `D3Correction.from_calculator` both recurse over this.

### D3 handling

If the NEB was converged on MACE+D3, **the virial sums over every energy term, exactly like
forces do.** D3 responds to affine deformation, so it carries a real virial. `D3Correction`
threads it into four places:

| quantity | treatment |
|---|---|
| energies / barriers | `+ E_D3` |
| virial `W` | `+ V·σ_D3` (read from the ASE calculator; exact first derivative) |
| `fmax` gate | `+ F_D3` |
| eigenvalue gate | `+ H_D3` (finite-differenced, 6N D3 calls) |
| per-atom `dw` | **MACE only** — flagged `dw_is_mace_only` |

Reading `σ_D3` gives exact **first** derivatives only. Second derivatives (`𝒞`) would need D3
inside the autograd graph or a finite-difference of `σ_D3`.

## 5. Invariants — every assertion exists for a reason, do not remove

```python
allclose(W[k], W[k].T)                 # hook symmetrises; failure = strain never applied
W.abs().max() > 0                      # zero virials = compute_stress not set
allclose(W_mace, V * out["stress"])    # independent code path inside MACE
allclose(sum_a w_atom, W_mace)         # atomic virial sum rule (MACE part only)
allclose(A_fwd - A_rev, A_rxn)         # vacuous but catches index/sign typos
n_negative_eigenvalues == (0, 1, 0)    # for (initial, saddle, final) — HARD GATE
```

`fmax` is recomputed inside `barrier_sensitivities` rather than trusted from the NEB, and
uses **raw** forces — ASE zeroes constrained components by default, and the envelope error
scales with the true residual.

## 6. Corrections already made — do not reintroduce

1. **Voigt factor of 2.** The **strain** vector gets the 2 (`γ_xy = 2η_xy`); the **gradient**
   converts by **plain copy**, like stress — which is what `A` is, since `A = V·σ`. An earlier
   draft had this backwards.
2. **Slabs.** `V·σ` reconstructs the virial **exactly** — vacuum padding cancels identically,
   since MACE *defines* `stress = virial/V`. Only `σ` **alone** is vacuum-dependent. Both
   arithmetic paths agree; the rule is about what you *report*.
3. **D3 has a real gradient.** An earlier phrasing ("needs no autodiff") was ambiguous; it
   meant "no autodiff *you* have to write", since torch-dftd computes it internally.

## 7. Architectural constraints

- **Never enable cuEq / OEQ / `torch.compile`** on the differentiable model. Source comment:
  "oeq ops are opaque to AOTAutograd." Fused kernels may not support double backward.
- Both calculator wrappers can hold the **same** `nn.Module`; `.to(dtype=...)` is in-place,
  so converting one converts the other. Don't do it mid-NEB.
- float64 everywhere. float32 makes `gradcheck` fail and caps `fmax`.
- If you ever add a TorchSim NEB: its batching assumes **independent** systems, so
  `autobatcher` must be off, `InFlightAutoBatcher` will pop images out of the band, and
  per-system convergence is meaningless. Also never wrap band positions (breaks tangents);
  use minimum-image convention.
- `MaceTorchSimModel.forward` **detaches** energy, forces, and stress — it cannot be used for
  autodiff without patching.

## 8. Open items

1. **Run the validation ladder** (below). Nothing has been executed yet.
2. Confirm D3 `damping`/`xc`/`cutoff` match what the NEB used — a mismatch is silent and
   surfaces only as inflated `fmax`.
3. `_assert_consistent` requires identical cells across images → fixed-cell, strain-controlled
   only. Variable-cell NEB is not supported.
4. `E_rxn` is labelled 0 K internal energy, **not** free energy (no entropy, no ZPE). The
   stress coupling for the driving force is the transformation strain `η_t`, which needs
   variable-cell endpoint relaxation and is not implemented.
5. Not implemented: activation elastic tensor `𝒞`, symmetry-variant enumeration for mechanism
   crossover, electric-field coupling.

## 9. Validation ladder — run in this order

1. `fd_check_virial` — central-differences a real affine deformation against the analytic
   virial. Tests the hook, symmetrisation, and signs. Three single-point energies.
2. `fd_check_d3_virial` — same for D3. Catches sign/units errors in the ASE-stress → virial
   handoff. **The most likely failure in the D3 addition.**
3. `n_negative_eigenvalues` on all three geometries — must be `(0, 1, 0)`.
4. `barrier_sensitivities` — check the `fmax` warnings before reading any number.
5. **Tier-3 (the only real test of the envelope assumption):** re-converge full CI-NEBs at
   `η ± h`, `h ≈ 1e-3`, and compare `(ΔE‡(+h) − ΔE‡(−h))/2h` against `A_fwd` contracted with
   that strain direction. Expect agreement to a few tenths of a percent.

Note steps 1–4 test plumbing. Only step 5 tests the physics.
