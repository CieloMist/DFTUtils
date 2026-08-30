# Handoff brief — extending strain sensitivity to finite-temperature nucleation

Companion to `HANDOFF.md` (single-saddle, 0 K) and `CONSTRAINED_SUBSPACE.md` (why the gates
run on the free subspace). Preview in VSCode (`Ctrl+Shift+V`) to render the math.

Paste this into a new chat as project context. **Confidence is labelled throughout**:
statistical-mechanics identities are standard results; method recommendations are judgement;
one section is explicitly research-level.

---

## 1. Goal

Strain sensitivity of **nucleation** in solid–solid phase transitions. These transitions are
often too complex to follow a single minimum energy path, so the single-saddle machinery
(validated, see §2) has to generalise to finite temperature and to a *distribution* of paths.

The requirement driving every choice below: **keep autodifferentiation**. The whole approach
rests on getting $\partial E / \partial \eta$ cheaply and exactly from the potential.

## 2. What already exists and works — do not rebuild

Module: `DFT_Utilities/BarrierSensitivity.py`. Driver: `Python_Scripts/MACE_Barrier_Sensitivity.py`.

At 0 K, for three converged stationary points (initial, saddle, final), the envelope theorem
gives every first-order output as a difference of virials $W = \partial E/\partial\eta = V\sigma$:

$$A_\text{fwd} = W_\text{saddle} - W_\text{initial}, \qquad A_\text{rxn} = W_\text{final} - W_\text{initial}$$

Second order needs implicit differentiation (the envelope theorem does **not** extend):

$$\frac{d^2F}{d\eta^2} = E_{\eta\eta} - E_{\eta u} H^{-1} E_{u\eta}$$

**Status: fully validated.** Tier-3 test (re-converged NEBs at $\eta = \pm 0.1\%$) agreed with
the analytic $A_\text{fwd}$ to **0.151%**. The activation elastic tensor reproduced
finite-difference $C_{yyyy}$ to 0.1–0.3%. D3 dispersion is inside the autograd graph
(`calc_energy_batch`), giving exact first *and* second derivatives.

Reference numbers for the test system (H hop in a Sn slab, 92 atoms, 32 free):
barrier 0.3664 eV; $A_{\text{fwd},yy} = -1.622$ eV per unit strain;
$C_{\text{act},yyyy} = 27.7$ eV per unit strain$^2$, of which **77% is internal relaxation**,
not clamped-ion.

**The key code hook for the new work:** `energies_and_virials(model, batch, n_systems=N)` is
already written for arbitrary $N$ — `barrier_sensitivities` merely passes 3. Feed it $N$
sampled configurations and average. One batched forward, one backward.

## 3. The central identity  [standard statistical mechanics]

For $F(\eta) = -k_BT \ln Z(\eta)$ with $Z = \int dR\, e^{-\beta E(R,\eta)}$:

$$\boxed{\ \frac{dF}{d\eta} = \left\langle \frac{\partial E}{\partial \eta} \right\rangle = \langle W \rangle\ }$$

One line: $d/d\eta$ hits the Boltzmann factor, brings down $-\beta W$, and the $-k_BT$ cancels
the $\beta$. **The strain derivative of a free energy is the ensemble-averaged virial.** No
stationary point, no envelope theorem — exact for any converged ensemble.

For a free energy profile along an order parameter $\lambda$, the same manipulation gives
$dF(\lambda)/d\eta = \langle W\rangle_\lambda - \langle W\rangle$, and the unconditional term
is $\lambda$-independent, so it cancels in any difference:

$$\frac{d\,\Delta F^\ddagger}{d\eta} = \langle W\rangle_\text{TS} - \langle W\rangle_\text{basin}$$

Structurally identical to the 0 K result, with each configuration promoted to an ensemble and
each virial to a conditional average.

### Second order at finite $T$  [standard]

$$\frac{d^2 F}{d\eta^2} = \left\langle \frac{\partial^2 E}{\partial\eta^2}\right\rangle - \beta\, \mathrm{Var}(W)$$

the fluctuation formula for elastic constants. Note the parallel to the 0 K expression in §2:
a clamped term minus a positive-definite softening. At 0 K the softening comes from implicit
differentiation of $\nabla_u E = 0$; at finite $T$ it comes from the **variance of the virial**.
Same physics, two languages, agreeing in the classical harmonic limit. Since relaxation was
77% of $C_\text{act}$ at 0 K, expect the fluctuation term to dominate — and it is far harder
to converge statistically than a mean.

## 4. What the ensemble formulation buys

- **Both 0 K gates disappear.** The free-atom `fmax` gate and the $(0,1,0)$ eigenvalue gate
  exist *only* because the 0 K envelope theorem needs $\nabla_u E = 0$. The ensemble identity
  needs nothing of the kind.
- **Genuine free energy**, including entropy. The 0 K `E_rxn` is explicitly labelled internal
  energy with no entropy or ZPE; for nucleation that is not a detail.
- **The cluster finally helps.** A single autograd graph is one process — 64 nodes bought
  nothing for the 0 K work. Ensemble averaging is embarrassingly parallel.

What replaces the gates: **ergodicity**, which is much harder to verify. Budget a
block-averaging or autocorrelation diagnostic on $\langle W\rangle$ to play the role the gates
played. The failure mode is a plausible-looking number from an unconverged ensemble.

## 5. Rates and forward flux sampling  [RESEARCH-LEVEL — treat with caution]

A rate is not a free energy, so $d\ln k/d\eta$ splits in two:

- **Static** — Boltzmann weights of configurations at each interface. Handled by §3.
- **Dynamic** — the path measure itself. For overdamped Langevin the Onsager–Machlup action
  depends on the forces, so $\partial S/\partial \eta$ brings in
  $\partial^2 E / \partial R\, \partial \eta$ — *the same mixed derivative already computed for
  $C_\text{act}$*. The machinery transfers; a working estimator does not come for free.

**Recommendation: do not make FFS the primary tool.** Its great advantage is needing only a
monotonic progress coordinate rather than a good reaction coordinate — but its output is a
rate, and computing only the static part discards that advantage while paying FFS's cost.
Use it to *validate* a rate obtained from the free-energy route plus a prefactor estimate;
disagreement tells you the prefactor is strain-sensitive, which is exactly the assumption the
cheap route rests on.

## 6. Sampling method — recommendation and rationale  [judgement]

### Umbrella sampling / ABF along nucleus size, with replica exchange — start here

- The identity applies **per window** with no modification: an umbrella window *is* a
  conditional ensemble.
- Windows are independent → embarrassingly parallel.
- **Correlated sampling across strain states is trivial** (identical windows, identical seeds,
  shared restart configurations). This matters more than it sounds:
  $\langle W\rangle_\text{TS} - \langle W\rangle_\text{basin}$ is a difference of large averages,
  the same cancellation that made $A_\text{rxn}$ unresolvable at 15% in the 0 K work. Variance
  cancellation is the difference between a resolved number and noise. **Design it in from the
  start**; do not sample the two ensembles independently.
- Gives $dF(N)/d\eta$ across the whole profile, so you also get how the critical nucleus size
  shifts with strain, $dN^*/d\eta$ — often the more interesting quantity.

### Is umbrella sampling shape-agnostic? — yes, formally

Sampling along $N$ does not *impose* a shape, it **marginalises** over shapes:
$F(N) = -k_BT\ln P(N)$ already contains the shape entropy. That is the correct thermodynamic
object.

The real problem is **ergodic, not definitional**. If shape relaxation is slow compared to
window sampling — and for solid–solid it usually is, since changing a habit plane is a
collective rearrangement — the window never explores shapes, and $F(N)$ is biased by a hidden
slow coordinate.

**Replica-exchange umbrella sampling is the single largest fix**: swaps let configurations
diffuse along $N$ carrying shape information between windows. Add multiple walkers per window
seeded from both growth and shrinkage trajectories.

Caveat: the sensitivity inherits any bias, possibly worse than the barrier does. Habit planes
exist *because* certain shapes minimise elastic energy, so $\langle W\rangle_N$ should be
strongly shape-dependent.

### The decision procedure is the committor test, not intuition

Harvest configurations at the putative barrier top, fire short unbiased trajectories,
histogram the committor. Peaked at 0.5 → $N$ is a genuine reaction coordinate, stop. Bimodal
or shifted → something is missing, and *now* you know shape is needed, empirically, at a
fraction of the cost of assuming it.

### Finite-temperature string — if the committor test fails

**FTS requires you to choose CVs.** The string cannot discover its own space: the transition
tube is Voronoi cells around string images *in CV space*, so the tessellation, restraints and
string update are all defined there. What the string gives free is the **path through** that
space — you need not guess the reaction coordinate or mechanism, only the space.

> Choose CVs that **span** the slow motions, not CVs that are individually good reaction
> coordinates. **Completeness, not correctness.** Missing CVs bias; extra CVs only cost
> sampling. Practical ceiling is roughly 3–6 dimensions.

Candidates: nucleus size; gyration-tensor eigenvalue ratios (shape/anisotropy); habit-plane
normal from the principal axis; transformation strain (**but see §7**).

If guessing still feels uncomfortable: learned CVs (TICA, diffusion maps, autoencoders on
short unbiased runs), or committor-guided refinement — find where the committor is poorly
predicted and add a descriptor correlating with the residual.

### Other methods

- **TPS + committor analysis**: use to validate that a CV is a genuine reaction coordinate,
  not for production sensitivity.
- **Metadynamics**: workable with reweighting, but nucleation free energies are steep and
  hysteresis-prone; no advantage over REUS here.

## 7. Traps — settle these before generating expensive data

1. **$\eta$-dependent order parameter.** If $\lambda = \lambda(R;\eta)$ explicitly, the
   derivative also hits the delta function and there is an extra term beyond
   $\langle W\rangle_\lambda - \langle W\rangle$. This is *most* tempting for displacive
   transitions, where transformation strain is the natural order parameter. **Fix: define
   $\lambda$ in scaled/fractional coordinates so it carries no explicit $\eta$ dependence.**

2. **Strain control vs stress control.** The current framework is fixed-cell, and
   $dF/d\eta = \langle W\rangle$ is the strain-controlled statement. But fixing the cell
   prevents a nucleus from accommodating its own transformation strain — inflating the barrier
   and amplifying image interactions. Under stress control the conjugate identity is equally
   clean ($G = F - V\sigma{:}\varepsilon$, with $dG/d\sigma$ an ensemble-averaged strain), but
   **the observable changes**. For "nucleation under load" the Gibbs version is usually the
   physically right question. Decide before choosing NVT vs NPT.

3. **Finite-size elastic image interactions — probably the largest error source.** A critical
   nucleus carries a strain field decaying as $1/r^3$, so in a periodic cell it interacts with
   its own images at long range, and nucleation barriers converge slowly with cell size. This
   can exceed the strain sensitivity being measured. **Budget a size-scaling study as
   first-class work, not a final check** — no sampling method diagnoses it for you. It is the
   finite-$T$ analogue of the free-region-size caveat in `CONSTRAINED_SUBSPACE.md`: every
   internal check passes happily on a too-small cell.

4. **Symmetry-related variants are discrete.** A continuous CV interpolating between them
   passes through unphysical intermediates. Run a separate string (or window set) per variant
   and compare. This is also how to address the mechanism-crossover item in `HANDOFF.md` §8.5 —
   crossover is a non-analytic kink in $\Delta E^\ddagger(\eta)$ that no Taylor order captures,
   and it is what actually bounds the strain range over which any of these numbers describe the
   real system.

## 8. Suggested sequencing

1. Umbrella sampling + replica exchange along nucleus size, fixed cell, at the existing strain
   states. Reuses the validated virial machinery essentially unchanged; fastest route to a real
   number.
2. **System-size scaling study** — decides whether the number means anything.
3. Committor test at the barrier top → decides whether shape CVs are needed.
4. Revisit stress vs strain control with data in hand.
5. FTS only if step 3 says so, with CVs informed by the harvested configurations rather than
   guessed.

## 9. Open questions for the new chat

- Which transition, and is it displacive/martensitic or reconstructive? This changes the CV
  choice and whether variants dominate.
- Temperature range, and whether quantum/ZPE corrections matter for the light species.
- Is the target observable a free-energy barrier or an actual rate? (§5 — the answer decides
  whether FFS is worth its cost.)
- Cell size accessible with MACE at the required sampling length — sets whether §7.3 is a
  solvable problem or a limiting one.
