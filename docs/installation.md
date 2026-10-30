---
title: Installation
---

# Installation

NIRSPipe needs Python 3.10 or later and is installed from source:

```bash
git clone https://github.com/AU-CONNECT-Lab/NIRSPipe.git
cd NIRSPipe
pip install -e ".[dev]"
```

Every dependency below is required and is resolved by the install. Check it with:

```bash
nirspipe --version
```

## bids-validator

`nirspipe`, `nirspipe-prep` and `nirspipe-qc` check their BIDS input with bids-validator when it is installed, and carry on with a warning when it is not. It runs on [Deno](https://deno.com):

```bash
deno install -A -g -n bids-validator jsr:@bids/validator
```

## GPU

`nirspipe-hyper` and `nirspipe-hyper-pairnull` can run the wavelet transforms on an NVIDIA card with `--device cuda` ([Running on a GPU](hyperscanning.md#running-on-a-gpu)). The optional extra installs cupy with the CUDA 12 libraries it needs, so no CUDA toolkit is required, only a driver for CUDA 12 or later:

```bash
pip install -e ".[gpu]"
```

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `mne` | ≥ 1.6 | fNIRS signal processing |
| `mne-nirs` | ≥ 0.7 | fNIRS preprocessing, GLM |
| `mne-bids` | ≥ 0.14 | BIDS read and write |
| `pybids` | ≥ 0.16 | BIDS dataset querying |
| `nilearn` | ≥ 0.10 | GLM design matrix and fitting |
| `statsmodels` | ≥ 0.14 | The robust refit of `--noise-model ar_irls`, and the cohort tests' p corrections |
| `nibabel` | ≥ 3.2 | Surface curvature for the brain renders |
| `scipy` | ≥ 1.11 | Filters, spectra |
| `numpy` | ≥ 1.24 | Numerics |
| `pandas` | ≥ 2.0 | Table I/O |
| `h5py` | ≥ 3.9 | SNIRF read and write |
| `plotly` | ≥ 5.18 | Interactive QC figures |
| `pyvista` | ≥ 0.43 | Off-screen 3D brain renders |
| `matplotlib` | ≥ 3.8 | Static QC panels |
| `pillow` | ≥ 10.0 | Stitches the rendered brain views side by side |
| `jinja2` | ≥ 3.1 | HTML report templates |
| `joblib` | ≥ 1.3 | Parallel `--nprocs` dispatch |
| `threadpoolctl` | ≥ 3.2 | Numeric threads per subject under `--nprocs` |
| `PyWavelets` | ≥ 1.4 | Wavelet motion correction |
| `pycwt` | ≥ 0.4 | Wavelet transform coherence for hyperscanning |
| `bibtexparser` | ≥ 2.0.0b1 | Methods citations |
| `tomli` | ≥ 2.0 | TOML reader (Python < 3.11 only) |
| `flask` | ≥ 3.0 | GUI (`nirspipe-gui`) and rating viewers (`nirspipe-rate`) |

`tables` (PyTables) is not among them. The package writes no HDF5 file of its own beyond SNIRF; install it yourself to call `save()` on a GLM object the Python API returned.

## Building these docs

The documentation is built with Sphinx. The usage page renders every command's flags from its own parser, so the package has to be installed first:

```bash
pip install -e .
pip install -r docs/requirements.txt
python -m sphinx -E -b html docs docs/_build/site
```
