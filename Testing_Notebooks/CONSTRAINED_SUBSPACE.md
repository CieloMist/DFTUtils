# Why the gates run on the free subspace

Companion note to `HANDOFF.md`. Preview in VSCode (`Ctrl+Shift+V`) to render the math.

Context: the `[100]` H-diffusion slab has 92 atoms, 60 of them held by `FixAtoms`, leaving
32 free. Both convergence gates in `MACE_Barrier_Sensitivity.py` operate on the free
coordinates only. This note says why that is exact rather than a convenience.

---

## 1. The partition

Split the $3N = 276$ coordinates into free and fixed:

$$R = (u, w), \qquad u \in \mathbb{R}^{96}\ \text{(free)}, \qquad w \in \mathbb{R}^{180}\ \text{(fixed at } w_0)$$

Define the **reduced potential energy surface**:

$$\tilde{E}(u) \;\equiv\; E(u,\, w_0)$$

This is not an approximation. It is the exact restriction of $E$ to the affine subspace
$\{w = w_0\}$, and it is a perfectly good 96-dimensional PES. The frozen atoms enter as
*parameters* — a static environment the free atoms feel and that never responds.

For a many-body potential like MACE you cannot literally split $E$ into free–free plus
free–fixed contributions, but that does not matter: $\tilde{E}(u)$ is well defined
regardless, and the frozen-region contribution is a constant that cancels in every energy
*difference*. That is why the barrier is meaningful even though the total energy is
dominated by the frozen bulk.

## 2. Gradient

$$\nabla_u \tilde{E} = \left.\frac{\partial E}{\partial u}\right|_{(u,\, w_0)}$$

The free-atom forces of the full system *are* the gradient of the reduced PES — nothing is
lost by ignoring the rest.

And $\partial E / \partial w \neq 0$ is not an error. It is the force the constraint must
supply to hold $w$ in place: a Lagrange multiplier. In this system that is the
$\approx 0.15$ eV/Å sitting on atom 20. Reporting it as a convergence failure is a category
error, which is exactly what the original raw-force `fmax` gate did.

## 3. Hessian — why the submatrix is exact

$$\nabla^2_u \tilde{E} = H_{uu}$$

The full Hessian has the block structure

$$H = \begin{pmatrix} H_{uu} & H_{uw} \\ H_{wu} & H_{ww} \end{pmatrix}$$

The blocks $H_{uw}$ and $H_{ww}$ describe coupling to, and curvature along, directions the
system cannot explore. They do not appear in $\nabla^2_u \tilde{E}$ at all.

Taking a plain submatrix is legitimate — rather than requiring a projection with correction
terms — because **the constraint is affine**: $w = \text{const}$. Restriction and
differentiation commute, so no curvature-of-the-constraint terms arise.

> For a *nonlinear* constraint (`FixBondLength`, a curved `FixedPlane`) this fails: you would
> need a projected Hessian carrying extra terms in the constraint's own second derivatives.
> This is precisely why `_free_mask` decodes only `FixAtoms` and conservatively treats
> everything else as free.

So "minimum" and "saddle" are statements about $\tilde{E}$, and the $(0, 1, 0)$ rule is a
statement about the eigenvalues of $H_{uu}$ — a $96 \times 96$ matrix, not $276 \times 276$.

**Zero modes.** In an unconstrained system three eigenvalues vanish by translational
invariance. Here, translating only the free atoms stretches bonds to the frozen region and
costs energy, so there are no zero modes at all. The tolerance in
`n_negative_eigenvalues` is therefore uncritical.

**Symptom when this is got wrong.** Diagonalising the full $276 \times 276$ matrix at a
properly converged minimum (free-atom fmax $4 \times 10^{-4}$) reported **28 negative
eigenvalues** — all of them curvature along frozen directions that are not variables and
not at a stationary point.

## 4. Connection to the strain derivative

Under strain the fixed atoms are **not** held in absolute space. They deform affinely with
the cell, so $w$ is a *known function* of $\eta$:

$$w(\eta) = w_0 + w_0\, \mathrm{sym}(\eta)$$

Therefore

$$\frac{d \Delta E^\ddagger}{d\eta}
= \underbrace{\frac{\partial E}{\partial \eta}\bigg|_{\text{all atoms affine}}}_{\textstyle W,\ \text{the virial}}
\;+\; \underbrace{\frac{\partial E}{\partial u}}_{\textstyle =\,0\ \text{at a stationary point}} \cdot \frac{d u^*}{d \eta}$$

The first term is the virial. MACE's displacement hook applies the affine motion to *every*
atom, frozen ones included, which is exactly why $W$ is correct despite the constraints —
the frozen atoms' response to strain is real, accounted for, and belongs in $W$.

The second term is what the envelope theorem kills, and it requires only
$\nabla_u \tilde{E} = 0$ — free-atom fmax $\to 0$.

**One partition explains both gates:**

| gate | condition | lives in |
|---|---|---|
| stationarity (`fmax`) | $\nabla_u \tilde{E} = 0$ | $u$ |
| index (`n_negative_eigenvalues`) | signature of $H_{uu}$ | $u$ |

Neither says anything about $w$.

## 5. The caveat

Everything above is exact mathematics about $\tilde{E}$. Whether $\tilde{E}$ is the *right*
physics is a separate modelling judgement: freezing the bulk asserts those atoms would not
have moved much anyway.

If the frozen region sits too close to the migrating H, the barrier is contaminated — and
that is a convergence question (free-region size, slab thickness) that **none of these gates
can detect**. They will all pass happily on a too-small free region.
