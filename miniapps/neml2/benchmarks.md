# NEML2 miniapp — run log

Running notes on runs of the `neml2` solid-mechanics miniapp (branch `neml2-v3`,
NEML2 v3 cpp-aoti). **This is currently a correctness/convergence log, not a
benchmark: no wall-clock time has been measured yet** (see TODO). It will grow
into a benchmarking story as timing is added. All runs use the single CUDA build
(`build-cuda/miniapps/neml2/neml2`), which runs on both CPU (`-d cpu`) and GPU
(`-d cuda`).

## Setup (common to every run below)

- **Model:** `elasticity_aoti` — `LinearIsotropicElasticity` (E=100, ν=0.3),
  compiled by `neml2-compile` for cpu+cuda, float64. Strain/stress are SR2
  (Mandel). Constitutive evaluation uses `forward` (residual), `jvp` (matrix-free
  PA gradient action), and `jacobian` (assembled tangent).
- **Problem:** 3D unit cube, `-n 5` → 5×5×5 hex mesh, vector H1 displacement.
  `-gr 0 -or 1` (finest order 2). **3993 true DOFs.** Fixed face + prescribed
  x-displacement 0.001 on another face. Pseudo-time hard-coded t = 0.01.
- **Solver:** PETSc SNES (Newton) + CG (`-ksp_type cg`, `ksp_rtol 1e-4`,
  `ksp_max_it 30`, `snes_max_it 10`). PARTIAL assembly; the Jacobian is a PETSc
  `MATSHELL`; preconditioner is a geometric multigrid (HYPRE BoomerAMG on the
  coarse level + Chebyshev smoothers on finer levels). GPU runs use
  `-use_gpu_aware_mpi 0` (conda MPICH is not GPU-aware; transfers are host-staged).
- The material is linear, so the tangent is constant and Newton reaches its floor
  in 2 iterations; the initial residual ‖F‖₀ = 1.986928141239e-01 is identical in
  every run.

## Runs

| # | device | ranks | Newton its | ‖F‖ history | KSP its / step | result |
|---|--------|-------|-----------|-------------|----------------|--------|
| 1 | cpu  | 1 | 2 | 1.99e-1 → 7.84e-6 → 3.70e-10 | 5, 6 | converged (FNORM_ABS) |
| 2 | cpu  | 4 | 2 | 1.99e-1 → 9.38e-6 → 5.66e-10 | 5, 6 | converged (FNORM_ABS) |
| 3 | cuda | 1 | 2 | 1.99e-1 → 7.83e-6 → 3.78e-10 | 5, 6 | converged (FNORM_ABS) |
| 4 | cuda | 4 (2× RTX A5000) | 2 | 1.99e-1 → 8.94e-6 → 5.58e-10 | 5, 6 | converged (FNORM_ABS) |

Commands (from the MFEM repo root, env activated, `elasticity_aoti/` compiled,
`LD_LIBRARY_PATH` including `$GPU_TPLS/lib`):

```bash
              ./build-cuda/miniapps/neml2/neml2 -d cpu  -i elasticity_aoti/model   # 1
mpirun -np 4  ./build-cuda/miniapps/neml2/neml2 -d cpu  -i elasticity_aoti/model   # 2
              ./build-cuda/miniapps/neml2/neml2 -d cuda -i elasticity_aoti/model   # 3
mpirun -np 4  ./build-cuda/miniapps/neml2/neml2 -d cuda -i elasticity_aoti/model   # 4
```

## Observations

- All four configurations converge in 2 Newton iterations, each with the same KSP
  count per step (5 then 6), to ‖F‖ ≈ 1e-10. CPU and GPU agree to the
  linear-solver floor; the small differences in the 2nd/3rd residuals across runs
  come from parallel/GPU reduction ordering, not a behavioral difference.
- Run 4 spread the 4 MPI ranks round-robin over the 2 A5000s.

## Caveats / TODO before this is a benchmark

- **No timing measured.** Add `-log_view` (PETSc) and/or MFEM timers; report
  assembly, solve, and per-`forward`/`jvp`/`jacobian` NEML2 time separately.
- **3993 DOFs is tiny** — dominated by launch/setup overhead, not representative;
  GPU will look bad here. Scale `-n` and refinements up for meaningful numbers.
- Not yet varied: mesh size, assembly level (PARTIAL vs FULL), refinement counts,
  GPU-aware MPI, cpp-eager vs cpp-aoti, single- vs multi-GPU scaling.
