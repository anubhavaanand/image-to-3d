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

```bash
git clone https://github.com/traveller59/spconv.git
cd spconv
pip install .
```

**Requirements:**
- CUDA Toolkit installed and discoverable via `CUDA_HOME` or `PATH`
- PyTorch compiled for the same CUDA version
- Ninja build system (`pip install ninja`) can speed up compilation

---

## 3. `flash-attn` (TRELLIS attention backend)

TRELLIS defaults to `flash-attn` for efficient attention. Use a pre-built wheel
matching your CUDA and PyTorch versions whenever possible.

```bash
pip install flash-attn --no-build-isolation
```

**Alternative:** If `flash-attn` is unavailable for your CUDA/PyTorch combo,
install `xformers` instead and set `os.environ["ATTN_BACKEND"] = "xformers"`.

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu118
```

TRELLIS will auto-detect `flash-attn` first, then fall back to `xformers`.

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
export ATTN_BACKEND=flash-attn   # or 'xformers'
```

You can also set them at the top of your Python script before importing
`trellis`:

```python
import os
os.environ["SPCONV_ALGO"] = "native"
os.environ["ATTN_BACKEND"] = "flash-attn"

from trellis.pipelines import TrellisImageTo3DPipeline
```

---

## 7. Verified Configurations

| Framework | CUDA | PyTorch | Notes |
|-----------|------|---------|-------|
| LGM | 11.8 | 2.1.0 | Official tested config |
| TRELLIS | 11.8 / 12.2 | 2.4.0 | Official tested configs |
| TripoSR | 11.8+ | 2.1.0+ | Uses `torchmcubes` |

If you encounter build errors, check that your CUDA, PyTorch, and the extension
are all compiled for the **same** CUDA version.
