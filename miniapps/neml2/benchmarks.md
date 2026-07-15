# NEML2 miniapp — GPU plasticity solver benchmarking (hand-off)

Plan for the scaling study on the larger machine. The **fine operator is always
matrix-free** (cached consistent tangent applied as `dσ = M·dε`); algebraic
multigrid runs only on the assembled P1 coarse level of the p-multigrid. Build:
see `build.md`. Models compiled with `neml2-compile` into `*_aoti/` (gitignored).

Run e.g.:
```
mpirun -np <N> ./neml2 -d cuda -i chaboche_aoti/model -n 16 -or 1 -nt 5 \
       --profile --ksp fgmres --smoother chebyshev --coarse gamg
```

## Knobs (CLI)

| flag | values (default) | meaning |
|------|------------------|---------|
| `-i` | `<m>_aoti/model` | NEML2 cpp-aoti artifact |
| `-sv`/`-yv`/`-tv` | `strain`/`stress`/`t` | strain / stress / time variable names |
| `-n` | int (5) | elements per dimension (n³ hexes) |
| `-gr` / `-or` | int (0 / 1) | geometric (h) / order (p) MG levels; finest order 2^or |
| `-nt`/`-dt`/`-umax` | (5 / 0.01 / 0.001) | load steps / time increment / final displacement |
| `-d` | `cpu`\|`cuda` | device; use `mpirun` for MPI ranks |
| `--profile` | off | split constitutive vs linear-solve time + per-step Newton/KSP counts |
| `--ksp` | `cg`\|`gmres`\|`fgmres` (cg) | outer Krylov |
| `--smoother` | `chebyshev`\|`jacobi` (chebyshev) | matrix-free fine smoother |
| `--smoother-order` | int (2) | Chebyshev degree |
| `--coarse` | `boomeramg`\|`gamg`\|`none` (boomeramg) | assembled-coarse solver (`none` = single level) |
| `--coarse-mode` | `inner-cg`\|`vcycle` (inner-cg) | coarse solve: AMG-preconditioned CG vs one AMG V-cycle |
| `--tuned-amg` | off | plasticity-tuned scalar BoomerAMG params (PMIS, ext+i, aggressive coarsening) |
| `--near-null-space` | on | rigid-body modes for GAMG coarse |

Consistency: `--coarse-mode inner-cg` is a variable preconditioner → use `--ksp fgmres`
(the app warns otherwise); `vcycle` keeps `cg`/`gmres` valid.

## Models

| file | type | history (`~1`) |
|------|------|----------------|
| `elasticity.i` | linear elastic | none |
| `j2.i` | rate-independent J2, iso+kin hardening (KKT) | plastic/kin/eq strains, flow_rate, t |
| `chaboche.i` | Perzyna viscoplastic, 2 backstresses + Voce (rate form) | strain, stress, X1, X2, eq strain, t |

## Combinations to run (× each model, × cpu/cuda, × mesh size / ranks)

| id | `--ksp` | `--smoother` | `--coarse` (`--coarse-mode`) | extra | role |
|----|---------|--------------|------------------------------|-------|------|
| MG-CG    | cg     | chebyshev | boomeramg (inner-cg) | — | symmetric baseline |
| MG-BAMG  | fgmres | chebyshev | boomeramg (vcycle) | `--tuned-amg` | robust workhorse |
| MG-GAMG  | fgmres | chebyshev | gamg (vcycle) | `--near-null-space` (elastic) / `--no-near-null-space` (plastic) | native SA-AMG |
| MG-Jac   | fgmres | jacobi | boomeramg (vcycle) | `--tuned-amg` | cheap smoother (elastic; diverges for stiff j2) |
| single   | fgmres | chebyshev | none | — | no coarse grid (throughput ceiling) |

## Comparisons of interest (why)

- **CG vs FGMRES** — symmetric fast path vs robustness on the (often indefinite)
  plastic tangent; CG can stall/diverge as plasticity develops.
- **BoomerAMG vs GAMG** — coarse-AMG quality; also GPU maturity (hypre-CUDA vs
  GAMG) matters for a fully on-device solve.
- **tuned vs default BoomerAMG** — the tuned scalar params cut KSP ~30% on j2.
- **near-null-space on/off** — helps elasticity but hurts developed plasticity
  (rigid modes are the elastic near-kernel, not the plastic tangent's).
- **Chebyshev vs Jacobi smoother** — cost per apply vs robustness (Jacobi diverges
  in MG for stiff rate-independent j2).
- **MG vs single-level** — what the coarse grid buys (grows with problem size).
- **Per-model** — for viscoplastic chaboche the constitutive return-map dominates
  wall-clock (~85%), so the linear solver mainly moves KSP count, not time; for
  rate-independent j2 the linear cost is comparable, so preconditioner choice moves
  wall-clock. Report **KSP iterations** and the **constitutive-vs-linear time
  split** (`--profile`) separately.
- **Weak scaling** — fixed DOFs/rank as ranks and GPUs grow (lattice meshes to be
  provided); watch KSP-count growth and constitutive/linear balance.

## Notes

- CPU/CUDA KSP counts differ by a few iterations (AMG-setup nondeterminism, PMIS
  randomization) — compare within a device; the operator and solution are
  device-identical, and the FE↔NEML2 coupling is MOOSE-validated.
- Rate-independent j2's KKT return map can hit a singular local solve on GPU
  cuSOLVER at deep plasticity (`umax ≳ 0.02`); keep load steps moderate.
