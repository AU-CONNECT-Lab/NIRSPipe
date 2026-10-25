"""The coherence transform on a CUDA card: the CPU path's arithmetic, in the same order, in cupy.

::

  open_device("cuda")   a CudaDevice for one group's run, or None for "cpu"
  check_card(jobs)      refuses a run whose processes the card cannot hold
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from nirspipe.exceptions import DeviceError
from nirspipe.pipeline.hyper import wtc as _wtc
from nirspipe.utils.logging import get_logger

logger = get_logger("pipeline.wtc_gpu")

# Card memory one process's pool may hold. Fixed, so a shared card stays shared
POOL_LIMIT_GIB = 4
# What a process holds outside its pool: the CUDA context, kernels, cuFFT plans
_CONTEXT_GIB = 0.5
# Transforms stop being cached once the pool holds this share of its limit
_CACHE_SHARE = 0.5
_PLAN_CACHE_SIZE = 4
# The check judges cells above this power share, where round-off is far under a float32 ulp
_JUDGED_SHARE = 1e-6
# A null that pads its draws meets several record lengths; older kernels are dropped
_MAX_KERNEL_SETS = 2

INSTALL_HINT = 'pip install "nirspipe[gpu]"'


def _cupy():
    try:
        with warnings.catch_warnings():
            # cupy warns on every import when no CUDA toolkit is installed; the pip wheels
            # carry the libraries it needs
            warnings.filterwarnings("ignore", message="CUDA path could not be detected")
            import cupy
    except ImportError as exc:
        raise DeviceError(f"--device cuda needs cupy, which is not installed ({exc}). "
                          f"Install it with: {INSTALL_HINT}") from exc
    return cupy


def _card_properties(cp) -> dict:
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise DeviceError("--device cuda: no CUDA card is visible")
        return cp.cuda.runtime.getDeviceProperties(0)
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise DeviceError(f"--device cuda: no usable CUDA card ({exc})") from exc


def check_card(jobs: int) -> str:
    """The card's name, after refusing a run of ``jobs`` processes it cannot hold."""
    cp = _cupy()
    props = _card_properties(cp)
    total = props["totalGlobalMem"] / 2 ** 30
    need = jobs * (POOL_LIMIT_GIB + _CONTEXT_GIB)
    if need > total:
        raise DeviceError(
            f"--device cuda with {jobs} group(s) at once needs about {need:.1f} GiB of card "
            f"memory ({POOL_LIMIT_GIB} GiB per process plus its context), and the card has "
            f"{total:.1f} GiB. Lower --nprocs, or run on the CPU.")
    return props["name"].decode()


def open_device(device: str) -> "CudaDevice | None":
    """The card ``--device`` names for one group's run, None for the CPU."""
    if device == "cpu":
        return None
    if device == "cuda":
        return CudaDevice()
    raise ValueError(f"unknown device {device!r}; expected 'cpu' or 'cuda'")


def device_params(device: "CudaDevice | None") -> dict:
    """What a coherence sidecar records about the card, nothing for the CPU."""
    if device is None:
        return {}
    env = device.environment
    return {"wtc_device": "cuda", "wtc_device_name": env["gpu_name"],
            "wtc_cupy_version": env["cupy_version"],
            "wtc_cuda_version": env["cuda_runtime_version"]}


def _version(v: int) -> str:
    """CUDA's integer version as written: 12090 -> '12.9'."""
    return f"{v // 1000}.{(v % 1000) // 10}"


@dataclass
class _Kernels:
    """One grid's constants on the card, built once and shared by every channel and pairing."""
    psi: object
    time_filter: object
    scale_window: object
    scales: object
    n_fft: int
    n_sig: int


class CudaDevice:
    """One group's use of the card; ``check`` is None until a pairing was compared with the CPU."""

    def __init__(self) -> None:
        cp = self.cp = _cupy()
        props = _card_properties(cp)
        limit = POOL_LIMIT_GIB * 2 ** 30
        free, _ = cp.cuda.runtime.memGetInfo()
        if free < limit:
            raise DeviceError(
                f"--device cuda: the card has {free / 2 ** 30:.1f} GiB free and a process "
                f"needs {POOL_LIMIT_GIB} GiB; something else is using it. Wait, lower "
                "--nprocs, or run on the CPU.")
        self.pool = cp.get_default_memory_pool()
        self.pool.set_limit(size=limit)
        plans = cp.fft.config.get_plan_cache()
        plans.set_memsize(-1)
        plans.set_size(_PLAN_CACHE_SIZE)
        self.name = props["name"].decode()
        self.environment = {
            "gpu_name": self.name,
            "cupy_version": cp.__version__,
            "cuda_runtime_version": _version(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_version": _version(cp.cuda.runtime.driverGetVersion()),
        }
        self.check: "dict | None" = None
        self._kernels: dict = {}

    # ---- the three steps ----

    def _kernels_for(self, dt: float, grid) -> _Kernels:
        key = (round(dt, 12), int(grid.n_fft), int(grid.scales.shape[1]), len(grid.sj))
        if key not in self._kernels:
            while len(self._kernels) >= _MAX_KERNEL_SETS:
                self._kernels.pop(next(iter(self._kernels)))
            cp, n_fft, sj = self.cp, grid.n_fft, grid.sj
            # the same expressions as wtc._cwt and wtc._morlet's smooth, on the host, so the
            # constants are the CPU's to the bit before they move
            ftfreqs = 2 * np.pi * np.fft.fftfreq(n_fft, dt)
            self._kernels[key] = _Kernels(
                psi=cp.asarray((sj[:, None] * ftfreqs[1] * n_fft) ** 0.5
                               * np.conjugate(grid.mother.psi_ft(sj[:, None] * ftfreqs))),
                time_filter=cp.asarray(np.exp(-0.5 * (sj[:, None] / dt) ** 2
                                              * (2 * np.pi * np.fft.fftfreq(n_fft)) ** 2)),
                scale_window=cp.asarray(_wtc._scale_window(_wtc.WTC_DJ)),
                scales=cp.asarray(grid.scales), n_fft=n_fft, n_sig=grid.scales.shape[1])
        return self._kernels[key]

    def _smooth(self, X, k: _Kernels):
        """Along time by the wavelet's Gaussian, then along scale by the boxcar, as wtc's Morlet does."""
        from cupyx.scipy.ndimage import convolve1d

        cp = self.cp
        T = cp.fft.ifft(k.time_filter * cp.fft.fft(X, n=k.n_fft, axis=1), axis=1)[:, :k.n_sig]

        def along_scale(part):
            return convolve1d(part, k.scale_window, axis=0, mode="constant", origin=0)

        if not cp.iscomplexobj(X):
            return along_scale(T.real)
        return along_scale(T.real) + 1j * along_scale(T.imag)

    def prepare_channel(self, y: np.ndarray, dt: float, grid) -> "_wtc._ChannelWavelet":
        """``wtc._prepare_channel`` after the normalisation, its arrays left on the card."""
        cp = self.cp
        k = self._kernels_for(dt, grid)
        W = cp.fft.ifft(cp.fft.fft(cp.asarray(y), n=k.n_fft) * k.psi, axis=1)[:, :k.n_sig]
        # the CPU path refuses a transform pycwt would drop scales from
        n_nan = int(cp.isnan(W).all(axis=1).sum())
        if 0 < n_nan < W.shape[0]:
            raise ValueError("wavelet scales differ from the run's grid")
        S = self._smooth(cp.abs(W) ** 2 / k.scales, k)
        freqs, coi = _wtc._freqs_and_coi(grid, grid.sj, len(y), dt)
        return _wtc._ChannelWavelet(W=W, S=S, coi=coi, freqs=freqs,
                                    no_power=S < _wtc._POWER_FLOOR * S.max(axis=1, keepdims=True))

    def pair_from_prepared(self, p1, p2, dt: float, grid):
        """``wtc._pair_from_prepared`` on the card."""
        cp = self.cp
        k = self._kernels_for(dt, grid)
        W12 = p1.W * cp.conj(p2.W)
        phase = cp.angle(W12)
        W12 /= k.scales
        S12 = self._smooth(W12, k)
        WCT = cp.abs(S12) ** 2 / (p1.S * p2.S)
        WCT[p1.no_power | p2.no_power] = cp.nan
        return WCT, phase

    def trim_pair(self, WCT, aWCT, coi, freqs, n_sig, step, fmin, fmax, signif=None):
        """``wtc._trim_pair`` on the card, so only the trimmed map crosses the bus."""
        cp = self.cp
        order = np.argsort(freqs)
        freqs_s = freqs[order]
        band = _wtc._in_band(freqs_s, fmin, fmax)
        rows = cp.asarray(order[band])
        WCT_band = cp.asnumpy(WCT[:, :n_sig][rows][:, ::step].astype(cp.float32))
        phase_band = cp.asnumpy(aWCT[:, :n_sig][rows][:, ::step].astype(cp.float32))
        sig_band = None
        if signif is not None and np.ndim(signif) == 1 and len(signif) == len(freqs):
            sig_band = np.asarray(signif)[order][band].astype(np.float32)
        return WCT_band, freqs_s[band], coi[:n_sig][::step].astype(np.float32), sig_band, phase_band

    # ---- memory ----

    def has_room(self) -> bool:
        return self.pool.used_bytes() < _CACHE_SHARE * self.pool.get_limit()

    def release(self) -> None:
        self.pool.free_all_blocks()

    def raise_if_device(self, exc: Exception) -> None:
        """Re-raise ``exc`` as a DeviceError when the card, not the pairing, failed."""
        if isinstance(exc, DeviceError):
            raise exc
        if not type(exc).__module__.startswith(("cupy", "cupyx", "cupy_backends")):
            return
        if isinstance(exc, self.cp.cuda.memory.OutOfMemoryError):
            raise DeviceError(
                f"the card ran out of memory: this recording needs more than the "
                f"{POOL_LIMIT_GIB} GiB a process may hold. Run it on the CPU.") from exc
        raise DeviceError(f"the card failed: {exc}") from exc

    # ---- the check against the CPU ----

    def verify(self, cpu: tuple, gpu: tuple, share: np.ndarray,
               excluded: "np.ndarray | None", pairing: str) -> None:
        """Refuse the run unless the judged cells agree with the CPU's to one float32 ulp."""
        c_wtc, c_freqs, c_coi, _, c_phase = cpu
        g_wtc, g_freqs, g_coi, _, g_phase = gpu
        if (c_wtc.shape != g_wtc.shape or not np.array_equal(c_freqs, g_freqs)
                or not np.array_equal(c_coi, g_coi)):
            raise DeviceError(f"the card's map of {pairing} is not on the CPU's grid: "
                              f"{g_wtc.shape} against {c_wtc.shape}. Run on the CPU.")
        kept = np.ones(c_wtc.shape, dtype=bool)
        if excluded is not None and len(excluded) >= kept.shape[1]:
            kept &= ~np.asarray(excluded[:kept.shape[1]], dtype=bool)[None, :]
        # near the power floor the CPU's own value is round-off, so neither side is judged
        judged = kept & (np.asarray(share) >= _JUDGED_SHARE)
        nan_c, nan_g = np.isnan(c_wtc[judged]), np.isnan(g_wtc[judged])
        if not np.array_equal(nan_c, nan_g):
            raise DeviceError(
                f"the card's coherence is undefined in other cells than the CPU's for "
                f"{pairing} ({int((nan_c != nan_g).sum())} cells). Run on the CPU.")
        if not judged.any():
            return
        a, b = c_wtc[judged][~nan_c], g_wtc[judged][~nan_c]
        d_wtc = np.abs(a.astype(np.float64) - b)
        ulp_wtc = np.spacing(np.maximum(np.abs(a), np.abs(b))).astype(np.float64)
        pa, pb = c_phase[judged][~nan_c], g_phase[judged][~nan_c]
        d_phase = np.abs(pa.astype(np.float64) - pb)
        # the same angle either side of the branch cut is a turn apart
        d_phase = np.minimum(d_phase, np.abs(2 * np.pi - d_phase))
        ulp_phase = np.spacing(np.maximum(np.abs(pa), np.abs(pb))).astype(np.float64)
        over = (d_wtc > ulp_wtc) | (d_phase > ulp_phase)
        if over.any():
            raise DeviceError(
                f"the card's result for {pairing} differs from the CPU's by more than one "
                f"float32 unit in the last place at {int(over.sum())} of {a.size} cells "
                f"(coherence up to {d_wtc.max():.3g}, phase up to {d_phase.max():.3g} rad), "
                "so nothing is written from it. Run on the CPU.")
        self.check = {"device_check_pairing": pairing,
                      "device_check_cells": int(a.size),
                      "device_check_cells_one_ulp": int(((d_wtc > 0) | (d_phase > 0)).sum()),
                      "device_check_cells_near_floor": int((kept & ~judged).sum())}
        logger.info("card checked against the CPU on %s: %d cells, %d one float32 unit apart, "
                    "%d near the power floor not judged", pairing, a.size,
                    self.check["device_check_cells_one_ulp"],
                    self.check["device_check_cells_near_floor"])
