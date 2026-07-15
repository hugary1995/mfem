# NEML2 miniapp — GPU plasticity solver benchmarking plan

Goal: compare a small set of representative, promising **solver/preconditioner
combinations** for GPU plasticity with the `neml2` miniapp (branch `neml2-v3`,
NEML2 v3 cpp-aoti). This document is the plan and methodology; results tables get
appended as runs happen. The prior content (single-step / correctness lab notes)
has been retired — that work is done and validated (multi-step stateful loading,
three models, cached consistent tangent, torch single-threading; see git history
and the miniapp source comments).

## Non-negotiable constraint: the fine operator stays matrix-free

The end goal is GPU throughput, so the **fine-level Jacobian is never assembled**.
The Newton Jacobian is a PETSc `MATSHELL` whose action is the cached consistent
tangent contracted per quadrature point (`dσ = M·dε`, one `jacobian()` solve per
linearization, then pure mat-vecs). Everything below preserves that.

Consequences that shape the design:

- **Algebraic multigrid (BoomerAMG, GAMG, ML) cannot run on the fine operator** —
  its setup reads matrix rows. AMG is therefore confined to the **assembled P1
  coarse level** of a geometric *p*-multigrid; the fine order-2 operator is only
  ever applied, never assembled.
- **Fine-level smoothers are limited to diagonal + mat-vec methods**: point-Jacobi
  and Chebyshev (polynomial). ℓ1-Jacobi, Gauss–Seidel, ILU need matrix rows, so
  they are only available as *coarse* AMG-internal relaxation, not as the
  matrix-free fine smoother.
- **Flexible Krylov when the coarse solve is a variable preconditioner.** If the
  coarse level is solved by an inner Krylov (AMG-preconditioned CG), the overall
  preconditioner is nonlinear and the outer Krylov must be flexible (FGMRES/FCG).
  To keep plain CG/GMRES valid we fix the coarse solve to a set number of AMG
  V-cycles (linear preconditioner). Both modes are worth having.
- **CG is valid for the associative small-strain models** (j2, chaboche have a
  *symmetric* consistent tangent); GMRES/FGMRES is the robustness fallback
  (indefinite at high load, or future non-associative models). CG-vs-FGMRES is
  itself one of the comparisons.

## Solver combinations to keep (initial set)

All keep the fine operator matrix-free; multigrid is *p*-multigrid (order-2 fine →
order-1 coarse) unless noted; AMG runs only on the assembled P1 coarse level; the
coarse AMG is given the 6 rigid-body modes as its near-null-space.

| # | Krylov | fine smoother (matrix-free) | coarse solve (P1, assembled) | what it isolates |
|---|--------|-----------------------------|------------------------------|------------------|
| 1 | CG     | Chebyshev                   | BoomerAMG                    | symmetric reference (today's setup); fastest when the tangent is SPD |
| 2 | FGMRES | Chebyshev                   | BoomerAMG (tuned params)     | robust workhorse; handles nonsymmetry/indefiniteness |
| 3 | FGMRES | Chebyshev                   | GAMG (smoothed aggregation)  | BoomerAMG vs GAMG at the coarse level (else = #2) |
| 4 | FGMRES | point-Jacobi                | BoomerAMG                    | cheaper smoother; does Chebyshev's extra mat-vecs pay off on GPU? |
| 5 | FGMRES | point-Jacobi                | none (single level)          | baseline / GPU throughput ceiling; what the coarse grid buys |

Reading as axes: **1↔2** CG vs FGMRES; **2↔3** BoomerAMG vs GAMG; **2↔4**
Chebyshev vs Jacobi; **4↔5** multigrid vs single-level.

Tuned BoomerAMG parameters to use on the coarse level (from production plasticity
practice; also the config if a monolithic-AMG variant is ever added):

```
-pc_hypre_boomeramg_strong_threshold 0.7
-pc_hypre_boomeramg_interp_type       ext+i
-pc_hypre_boomeramg_coarsen_type      PMIS
-pc_hypre_boomeramg_agg_nl            4
-pc_hypre_boomeramg_agg_num_paths     2
-pc_hypre_boomeramg_truncfactor       0.4
# with GMRES: -ksp_gmres_restart 301
```

### Secondary axes (expand later)

Chebyshev degree and #smoother sweeps; number of *p* and *h* multigrid levels;
coarse-solve tightness (V-cycles vs inner-Krylov tolerance); GMRES restart; GPU
vs CPU; MPI ranks and GPUs (weak scaling on lattice meshes, provided later).

## What we measure

Per combination, per problem size / device / rank count:

- **Time to solution** per load step and total; and **KSP iterations per Newton
  step** (linear-solver efficiency).
- **Linear-solve time reported separately from constitutive time.** Plasticity's
  cost is dominated by the NEML2 return-map solves (residual + one cached
  tangent), which is *independent of the linear solver*; a fair solver comparison
  must factor it out. Instrument with timers around the tangent/residual eval vs
  the KSP solve, plus PETSc `-log_view`.
- **AMG setup vs apply time** (coarse level), and **peak GPU memory**.
- Later: **weak-scaling efficiency** (fixed DOFs/GPU as ranks grow).

Always record: model (`elasticity`/`j2`/`chaboche`), device, ranks/GPUs, DOFs,
load level, and convergence outcome.

## Results — initial run (small mesh, CPU + CUDA spot-check)

First pass with the instrumentation + solver switchboard in place. **Small problem
(`-n 5 -or 1` → 3993 DOFs, 2-level p-MG, `-nt 2`), so absolute times are
setup/launch-dominated, not scaling numbers — the comparable metric is KSP
iterations.** Matrix-free fine operator throughout; `ksp_max_it 100`; `--profile`
adds device syncs (inflates the total slightly). Combos:
- **c1** = cg + Chebyshev(2) + BoomerAMG(untuned) + inner-cg (≈ original)
- **c2** = fgmres + Chebyshev(2) + BoomerAMG(**tuned**) + vcycle  *(recommended)*
- **c4** = fgmres + **point-Jacobi** + BoomerAMG(tuned) + vcycle

### CPU (`-d cpu`), totals over 2 steps

| model | combo | Newton | KSP | solve s | constitutive s (resid / tan) | linear+other s | result |
|-------|-------|--------|-----|---------|------------------------------|----------------|--------|
| elasticity | c1 | 4 | 22 | 0.94 | 0.03 | 0.91 | converged |
| elasticity | c2 | 4 | 24 | 0.87 | 0.01 | 0.86 | converged |
| elasticity | c4 | 4 | 32 | 0.71 | 0.01 | 0.70 | converged |
| j2 (umax 0.02) | c1 | 16 | 420 | 15.9 | 7.1 (1.4 / 5.8) | 8.8 | converged |
| j2 (umax 0.02) | c2 | 16 | **294** | 13.4 | 7.3 (1.3 / 5.9) | 6.1 | converged |
| j2 (umax 0.02) | c4 | — | — | — | — | — | **diverged** (Jacobi too weak) |
| chaboche | c1 | 8 | 85 | 14.5 | 12.1 (3.1 / 9.0) | 2.4 | converged |
| chaboche | c2 | 8 | **70** | 14.6 | 12.6 (3.3 / 9.2) | 2.0 | converged |
| chaboche | c4 | 8 | 142 | 14.3 | 12.3 (3.1 / 9.1) | 2.1 | converged |

### CUDA (`-d cuda`), combo c2, totals over 2 steps

| model | Newton | KSP | solve s | constitutive s | note |
|-------|--------|-----|---------|----------------|------|
| elasticity | 4 | 36 | 0.26 | 0.04 | 3.4× faster than CPU c2 |
| chaboche | 8 | 90 | 1.34 | 1.09 | **~11× faster than CPU c2** (batched return map) |
| j2 (umax 0.015) | 12 | 219 | 1.05 | 0.61 | converged |
| j2 (umax 0.02) | — | — | — | — | NEML2 `FatalError`: singular local solve at step 2 |

### Single-level baseline (combo #5, `--coarse none`), CPU, KSP its over 2 steps

No coarse grid — just the matrix-free fine smoother as the preconditioner. Shows
what the coarse grid buys.

| model | #5 Jacobi | #5 Chebyshev | 2-level MG (c2) |
|-------|-----------|--------------|-----------------|
| elasticity     | 143 |  86 |  24 |
| j2 (umax 0.02) | 939 | 578 | 294 |
| chaboche       | 296 | 179 |  70 |

The coarse grid cuts KSP iterations ~2–6×; single-level Chebyshev beats
single-level Jacobi (stronger smoother). All single-level cases **converge** —
including j2, where the 2-level MG+Jacobi (c4) *diverged*: a too-weak V-cycle
smoother can amplify error, whereas plain FGMRES + Jacobi just converges slowly.
(At this tiny size, single-level can still win wall-clock on easy problems —
elasticity #5 0.52 s vs MG 0.87 s — by skipping coarse AMG setup; the MG's fewer
iterations pay off as the problem hardens/grows.)

### Coarse AMG: BoomerAMG vs GAMG (combo #2 vs #3), CPU, KSP its over 2 steps

Both matrix-free fine + FGMRES + Chebyshev; only the assembled-coarse AMG differs.
GAMG here is smoothed aggregation **without** an explicit rigid-body near-null-space
(a documented follow-up — see below).

| model | c2 BoomerAMG (tuned) | c3 GAMG (no near-null-space) |
|-------|---------------------|-----------------------------|
| elasticity     |  24 |  28 |
| j2 (umax 0.02) | 294 | **251** |
| chaboche       |  70 |  69 |

GAMG's smoothed aggregation is competitive-to-better than tuned BoomerAMG even
without the near-null-space (j2 251 vs 294). Attaching the 6 rigid-body modes (the
ordering-safe way: project them as coarse `ParGridFunction`s via
`VectorFunctionCoefficient`, orthonormalize the true-dof vectors, `MatSetNearNullSpace`
on the coarse Mat) is expected to widen GAMG's edge for elasticity/j2; it is the
main remaining refinement for combo #3.

### Findings

- **Tuned BoomerAMG + FGMRES cuts KSP iterations** vs untuned CG: j2 420→294
  (−30%), chaboche 85→70 (−18%); elasticity already trivial (22→24).
- **GAMG (SA) is competitive-to-better than tuned BoomerAMG** on the coarse level
  even without a near-null-space (j2 251 vs 294; chaboche 69 vs 70) — a promising
  coarse solver once rigid-body modes are added.
- **Where the linear solve actually costs**: for j2 the linear work (~6–9 s) is
  comparable to the constitutive cost (~7 s), so preconditioner quality moves
  wall-clock. For chaboche the constitutive cost dominates (~83–86 %), so KSP
  savings barely change the total — KSP count is the right comparison metric, not
  wall-clock, at this size.
- **Tangent re-evaluation dominates the constitutive cost** (chaboche: tangent
  ~9 s vs residual ~3 s), because the preconditioner rebuilds the tangent on both
  levels every Newton iteration. This is the top optimization target (frozen /
  reused tangent) for chaboche-type models.
- **GPU win scales with constitutive weight**: elasticity 3.4×, chaboche ~11×
  (the batched return-map solve is far faster on GPU). This is the payoff of the
  matrix-free-fine design + batched constitutive.
- **Smoother**: point-Jacobi is fine (and cheapest) for elasticity, converges for
  chaboche (142 vs 70 KSP), but **diverges for stiff rate-independent j2** —
  Chebyshev is the robust default; Jacobi only for well-conditioned cases.
- **j2 GPU fragility**: the rate-independent KKT/complementarity return map hits a
  singular local Jacobian on GPU (cuSOLVER) at `umax ≥ 0.02` (deep plasticity);
  it is fine at `umax ≤ 0.015` and on CPU (LAPACK tolerates it). Viscoplastic
  chaboche is robust. A model-selection consideration for the GPU study, not a
  solver issue. (The miniapp currently core-dumps on the NEML2 `FatalError`;
  catching it for a clean abort is a small follow-up.)
- CPU and GPU KSP counts differ (e.g. chaboche 70 vs 90); this is benign
  AMG-setup nondeterminism (PMIS randomization). Compare KSP counts within a device.

## Correctness (validated, details omitted)

We verified this offline and removed the throwaway harness: (1) CPU/CUDA operator
parity — residual and Jacobian action agree to machine precision and converge to
the same solution, so KSP-count device differences are benign; (2) physics — a
homogeneous uniaxial test matched MOOSE (same NEML2 model, small-strain total
formulation) to ~1e-5, validating the FE<->NEML2 coupling. (The j2 rate-independent
KKT return map can hit a singular local solve on GPU cuSOLVER at deep plasticity,
`umax≳0.02` — a model-robustness note, not a solver bug.)

## Models (unchanged from the correctness work)

| file | type | history (`~1` inputs) |
|------|------|-----------------------|
| `elasticity.i` | linear elastic | none |
| `j2.i` | rate-independent J2, Voce iso + linear kin (KKT) | plastic_strain, kinematic_plastic_strain, equivalent_plastic_strain, flow_rate, t |
| `chaboche.i` | Perzyna viscoplastic, 2 backstresses + Voce iso (rate form) | strain, stress, X1, X2, equivalent_plastic_strain, t |

All use plain `strain`/`stress`; compile with
`neml2-compile <m>.i --model model --device cpu cuda --dtype float64 --output-dir <m>_aoti -d stress:strain --example-batch-shape '(2,)'`.
Run single-threaded per rank (torch is pinned to 1 thread in `main`; parallelism
is MPI). `NEML2_LOGS='newton=debug'` prints the internal return-map convergence.
