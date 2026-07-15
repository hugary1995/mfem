# Building the NEML2 miniapp (NEML2 v3, CUDA end-to-end)

Reproducible setup for the `neml2` solid-mechanics miniapp on the `neml2-v3`
branch, targeting **GPU**: the FE kernels, the HYPRE AMG, PETSc, and the NEML2
constitutive model all run on device. The same build also runs on CPU (`-d cpu`).

Everything lives in one conda env (`mfem`) and is built for **CUDA 12.6** — the
version the `torch==2.13.0+cu126` wheel bundles. This matters because the miniapp
bridges MFEM device memory to torch via `at::from_blob` (zero-copy), so MFEM,
HYPRE, PETSc and torch share device pointers in one process and must agree on the
CUDA runtime. HYPRE and PETSc are built from source with CUDA; METIS is CPU-only
(partitioning).

| Component | Version |
| --- | --- |
| python | 3.14 |
| gcc / g++ | 13.4 |
| cmake / ninja | 4.4 / 1.13 |
| mpich | 4.3 |
| cuda-toolkit | 12.6 |
| torch | 2.13.0+cu126 |
| neml2 | 3.0.7 |
| HYPRE (from source) | v2.32.0, `sm_86` |
| PETSc (from source) | v3.24.5, CUDA `sm_86` + HYPRE |

Paths below: the MFEM repo root is `$MFEM` (this repo); the NEML2 source checkout
is `$NEML2_SRC`; the CUDA third-party libraries install to `$GPU_TPLS`.

```bash
export MFEM=$PWD                       # run from the MFEM repo root
export NEML2_SRC=$MFEM/../neml2
export GPU_TPLS=$HOME/projects/gpu-tpls
```

## 1. Create and activate the env

```bash
conda create -y -n mfem -c conda-forge \
  python=3.14 cxx-compiler c-compiler fortran-compiler \
  cmake ninja make pkg-config mpich metis \
  cuda-toolkit=12.6 cuda-nvtx-dev=12.6
conda activate mfem
```

All commands below run in this activated env.

## 2. PyTorch (CUDA 12.6)

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
```

## 3. NEML2 v3 from source

NEML2's C++ runtime must compile against the torch you run, so torch + `nmhit`
are pre-installed and build isolation is off:

```bash
pip install nmhit scikit-build-core cmake ninja
CUDA_HOME=$CONDA_PREFIX CUDACXX=$CONDA_PREFIX/bin/nvcc \
  pip install "$NEML2_SRC" --no-build-isolation -v
python -c "import neml2, torch; print(neml2.__version__, torch.__version__)"
```

## 4. HYPRE with CUDA

```bash
mkdir -p $GPU_TPLS/src && cd $GPU_TPLS/src
git clone --depth 1 --branch v2.32.0 https://github.com/hypre-space/hypre.git

cmake -S hypre/src -B build/hypre -GNinja \
  -DCMAKE_INSTALL_PREFIX=$GPU_TPLS \
  -DHYPRE_WITH_CUDA=ON -DHYPRE_CUDA_SM=86 \
  -DHYPRE_ENABLE_SHARED=ON -DHYPRE_WITH_MPI=ON \
  -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=$CXX
ninja -C build/hypre install
```

## 5. PETSc with CUDA + HYPRE

```bash
cd $GPU_TPLS/src
git clone --depth 1 --branch v3.24.5 https://gitlab.com/petsc/petsc.git
cd petsc

L=$CONDA_PREFIX/targets/x86_64-linux/lib
python3 ./configure --prefix=$GPU_TPLS \
  --with-cc=mpicc --with-cxx=mpicxx --with-fc=0 \
  --with-cuda=1 --with-cudac=$CONDA_PREFIX/bin/nvcc \
  --with-cuda-include=$CONDA_PREFIX/targets/x86_64-linux/include \
  --with-cuda-lib="[$L/libcudart.so,$L/libcublas.so,$L/libcublasLt.so,$L/libcusparse.so,$L/libcusolver.so,$L/libcurand.so,$L/libcufft.so,$L/stubs/libcuda.so]" \
  --with-hypre-dir=$GPU_TPLS \
  --with-blaslapack-dir=$CONDA_PREFIX \
  --with-debugging=0 COPTFLAGS=-O2 CXXOPTFLAGS=-O2
make -j32 PETSC_DIR=$PWD PETSC_ARCH=arch-linux-c-opt all
make PETSC_DIR=$PWD PETSC_ARCH=arch-linux-c-opt install
```

The CUDA libraries are listed explicitly by full path via `--with-cuda-lib`.

## 6. MFEM + the miniapp with CUDA

```bash
cd $MFEM
export LD_LIBRARY_PATH=$GPU_TPLS/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
NEML2_ROOT=$(python -c "import neml2,os;print(os.path.dirname(neml2.__file__))")

cmake -S . -B build-cuda -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
  -DMFEM_USE_MPI=YES -DMFEM_USE_METIS=YES \
  -DMFEM_USE_CUDA=YES -DCUDA_ARCH=sm_86 \
  -DCMAKE_CUDA_HOST_COMPILER=$CONDA_PREFIX/bin/mpicxx \
  -DMFEM_USE_PETSC=YES \
  -DHYPRE_DIR=$GPU_TPLS -DMETIS_DIR=$CONDA_PREFIX \
  -DPETSC_DIR=$GPU_TPLS -DPETSC_ARCH= \
  -Dneml2_ROOT=$NEML2_ROOT
ninja -C build-cuda neml2
```

- `LD_LIBRARY_PATH` includes `$GPU_TPLS/lib` so the from-source `libpetsc`/`libHYPRE`
  are found (MFEM's PETSc detection compiles and runs a small test program).
- `CMAKE_CUDA_HOST_COMPILER=mpicxx`: the executable is linked by nvcc, and using
  the MPI wrapper as nvcc's host compiler carries MFEM's MPI into that link.

## 7. Compile the model, then run

The cpp-aoti route loads offline-compiled device kernels; compile once for both
devices (the CUDA path's Inductor step needs the toolkit headers on `CPATH`):

```bash
cd $MFEM/miniapps/neml2
CPATH=$CONDA_PREFIX/targets/x86_64-linux/include \
LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib \
  neml2-compile elasticity.i --model model --device cpu cuda --dtype float64 \
    --output-dir elasticity_aoti -d : --example-batch-shape '(2,)'
```

`-d :` compiles the derivative graphs (the miniapp uses `jvp` for the matrix-free
PA gradient action and `jacobian` for the assembled paths).

Run (`-i` is relative to `miniapps/neml2/`, pointing at the artifact root):

```bash
cd $MFEM
./build-cuda/miniapps/neml2/neml2 -d cuda -i elasticity_aoti/model
mpirun -np 4 ./build-cuda/miniapps/neml2/neml2 -d cuda -i elasticity_aoti/model
./build-cuda/miniapps/neml2/neml2 -d cpu  -i elasticity_aoti/model
```
