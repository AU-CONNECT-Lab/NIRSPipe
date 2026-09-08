"""Compute signal quality metrics (SQM) for fNIRS data.

Established metrics: SCI, PSP, CV, SNR, GVTD.
Experimental (may change or be removed): Cardiac Power (CP), per-chromophore gcor,
spike and motion-correction footprint, low-frequency drift.

One module per metric family, layered so the imports run one way::

    _helpers    shared plumbing: _safe_metrics, _mask_to_segments, long_short_channels
    gvtd        the global motion index, its threshold, and BAD_gvtd censoring
    coupling    SCI, PSP, CV, SNR, amplitude, channel distance, CP
    motion      per-channel spikes and the motion-correction footprint
    haemo       HbO-HbR correlation, CNR, band powers, gcor, drift, retention
    windowed    the same questions per window, on one shared grid
    aggregate   the dicts a run stores, assembled from the families above

Every name is re-exported here, so ``from fnirs_pipe.qc.metrics import X`` reaches any of
them and a caller need not know which module X is in.

Each compute_* function returns a flat dict and does no I/O. Assembling them into a
per-run record on disk is qc/sqm_record.py's job, not this package's.

Bad channels are handled by signal domain, not per function. The Beer-Lambert conversion
is the line:

    intensity / OD (compute_raw_sqm and its helpers, motion_correction_metrics)
        aggregates cover every channel, rejected ones included. A rejected channel is
        still part of what the machine recorded, and this is the archival view.
    haemoglobin (compute_haemo_sqm, compute_prep_haemo_sqm)
        aggregates exclude them, which is what mne.pick_types does by default. By this
        stage the channel is out of the analysis.

Do not "fix" an intensity/OD function to exclude bads. Doing so to _sci_metrics in
particular makes sci_mean an average over channels chosen for having good SCI, which
cannot fall below the threshold no matter how bad the recording is.

Per-channel dicts always list every channel, in both domains.
"""

from fnirs_pipe.qc.metrics._helpers import (  # noqa: F401
    SHORT_MAX_DIST, LONG_MIN_DIST, LONG_MAX_DIST, long_short_channels,
    separation_bands, unclaimed_separations, _mean_or_none,
    _safe_metrics, _mask_to_segments,
    SCI_PASS, PSP_PASS, CV_PASS, SNR_PASS,
)
from fnirs_pipe.qc.metrics.gvtd import (  # noqa: F401
    GVTD_MOTION_BAND, GVTD_N_STD, gvtd_timetrace, gvtd_threshold,
    gvtd_channel_picks, gvtd_channel_blocks, _motion_metrics, gvtd_censor_spans, _windowed_gvtd,
    compute_windowed_gvtd, compute_windowed_filtered_gvtd,
)
from fnirs_pipe.qc.metrics.screening import (  # noqa: F401
    CRITERIA, Criterion, criterion_cutoffs, resolve_cutoffs, screen_channels,
    screening_scores,
)
from fnirs_pipe.qc.metrics.coupling import (  # noqa: F401
    PSP_WINDOW_S, compute_sci_scores, compute_psp_scores, _sci_metrics,
    channel_cv, channel_snr, _intensity_metrics,
    _channel_distance_metrics, _psp_metrics, _cardiac_power_metrics,
)
from fnirs_pipe.qc.metrics.motion import (  # noqa: F401
    _spike_mask, _motion_band_diff, _spike_metrics, _correction_footprint,
    motion_correction_metrics, motion_corrected_segments, spike_segments,
)
from fnirs_pipe.qc.metrics.haemo import (  # noqa: F401
    CNR_BASELINE_S, CNR_RESPONSE_S, haemo_quality_metrics, _cnr_metrics, _spectral_metrics,
    _gcor, gcor_metrics, _drift_metrics, _retention_metrics,
)
from fnirs_pipe.qc.metrics.windowed import (  # noqa: F401
    compute_windowed_sci, compute_windowed_psp, attach_windowed_series,
)
from fnirs_pipe.qc.metrics.aggregate import (  # noqa: F401
    compute_raw_sqm, compute_haemo_sqm, compute_prep_haemo_sqm, comparable_stage_metrics,
    _band_power, _variance_remaining,
)
