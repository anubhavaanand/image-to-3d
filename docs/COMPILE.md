# CUDA Extension Compilation Guide

This project depends on several custom CUDA extensions that must be compiled
from source. The sections below cover each one.

> **Note:** All commands below assume a Linux environment with an NVIDIA GPU,
> CUDA Toolkit installed, and a working PyTorch installation whose CUDA version
> matches your system CUDA.

---

## 1. `diff_gaussian_rasterization` (required by LGM)

LGM uses a modified 3D Gaussian rasterizer for rendering and saving PLY files.

```bash
git clone https://github.com/ashawkey/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
pip install .
```

If the build fails, make sure you have a C++ compiler and that your PyTorch
CUDA version matches the system CUDA Toolkit version.

---

## 2. `spconv` (required by TRELLIS)

TRELLIS uses sparse convolution operations for its structured latent models.

> **Do not clone `traveller59/spconv` and `pip install .`** — that is the old recipe and it
> no longer builds on Python 3.13 / torch 2.11. Use the prebuilt wheels:

```bash
pip install spconv-cu126   # CUDA 12.6 wheels
# fallback if your mirror has no cu126 build:
pip install spconv-cu120   # CUDA 12.0 wheels
spconv` is also the only practical route to the `cumm` headers it JIT-compiles against;
set `CPATH` to the installed cumm include dir so the JIT build finds them:
```

```bash
export CUMM_HOME=$(python -c "import cumm, pathlib; print(pathlib.Path(cumm.__file__).parent)")
export CPATH="$CUMM_HOME/include:$CPATH"
```

**Requirements:**
- CUDA Toolkit installed and discoverable via `CUDA_HOME` or `PATH`
- NVIDIA GPU with compute capability 7.0+ (V100/T4 or newer)
- Ninja build system (`pip install ninja`) speeds up the cumm JIT
- Minor-version compatibility is enough: **cu126 wheels run fine under a cu128 PyTorch**
  (e.g. torch 2.11+cu128 with spconv-cu126), so you do not need to match the 12.8 exactly.

---

## 2.5 `kaolin` (required by TRELLIS, via a shim)

TRELLIS imports `kaolin`, but **there is no kaolin wheel for torch 2.11 / Python 3.13**.
The benchmark therefore installs a deliberate `sys.modules` shim in the notebook instead of
the real package. This is intentional, not an oversight — replacing the shim with a real
`pip install kaolin` will pull an incompatible torch and break everything downstream. The
shim must stay in place for as long as the benchmark runs on Python 3.13.

---

## 2.6 `utils3d` (required by TRELLIS)

TRELLIS pins `utils3d` to a specific commit — newer tags break the pipeline's signatures:

```bash
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
```

---

## 2.7 `FlexiCubes` (TRELLIS mesh extraction)

`FlexiCubes` is **not** a pip package in the working configuration — it lives as a git
submodule inside the TRELLIS repository and must be importable from there. Clone TRELLIS
with `--recursive` and keep that tree on `sys.path`.

---

## 3. `flash-attn` / `xformers` (TRELLIS attention backend)

**TRELLIS defaults to `xformers`, not `flash-attn`.** `src/engines/trellis_engine.py` sets
`os.environ.setdefault("ATTN_BACKEND", "xformers")` before the TRELLIS import, precisely
because `flash-attn` has no prebuilt wheel for the torch 2.11 / Python 3.13 stack and
building it from source fails there.

```bash
pip install xformers
```

If you genuinely have flash-attn available for your CUDA/PyTorch combo, you can opt back in:

```bash
pip install flash-attn --no-build-isolation
# then set ATTN_BACKEND=flash-attn in the environment before importing trellis
```

TRELLIS will **fall back** to `xformers` only if `flash-attn` is unavailable; it does not
auto-detect flash-attn first.

---

## 4. `diffoctreerast` (TRELLIS differentiable mesh rendering)

Used by TRELLIS for differentiable mesh extraction and rendering.

```bash
git clone https://github.com/facebookresearch/diffoctreerast.git
cd diffoctreerast
pip install .
```

**Requirements:**
- CUDA Toolkit 11.8 or 12.2 (tested configurations)
- PyTorch with matching CUDA version
- `nvdiffrast` must be installed first: `pip install nvdiffrast`

---

## 5. `nvdiffrast` (TRELLIS rendering utility)

```bash
pip install nvdiffrast
```

On some systems you may need to install from source:

```bash
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
pip install .
```

---

## 6. Environment Variables

Set these before importing TRELLIS to avoid runtime errors:

```bash
export SPCONV_ALGO=native        # Recommended for single-run inference
export ATTN_BACKEND=xformers     # flash-attn is NOT built for this stack (see §3)
```

You can also set them at the top of your Python script before importing
`trellis`:

```python
import os
os.environ["SPCONV_ALGO"] = "native"
os.environ["ATTN_BACKEND"] = "xformers"

from trellis.pipelines import TrellisImageTo3DPipeline
```

---

## 7. Verified Configurations

| Framework | CUDA | PyTorch | Python | Notes |
|-----------|------|---------|--------|-------|
| **Benchmark (all 3 engines)** | **12.8** | **2.11** | **3.13** | **working config — torch 2.11+cu128 with spconv-cu126, xformers, kaolin shim, utils3d pinned to `9a4eb15e`** |
| LGM | 11.8 | 2.1.0 | 3.10 | Official tested config |
| TRELLIS | 11.8 / 12.2 | 2.4.0 | 3.10 | Official tested configs |
| TripoSR | 11.8+ | 2.1.0+ | — | Uses `torchmcubes` |

The benchmark deliberately runs cu126 wheels under a cu128 PyTorch, which violates the
"same CUDA version" rule below: NVIDIA minor-version compatibility makes the cu126
spconv/xformers wheels safe on a 12.8 runtime, and matching 12.8 exactly is not possible
for these packages. Do not "fix" this by downgrading torch.

If you encounter build errors, check that your CUDA, PyTorch, and the extension
are all compiled for the same CUDA version — **unless** the combination above, which is
verified and takes precedence.
