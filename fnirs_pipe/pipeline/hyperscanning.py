"""Hyperscanning pipeline utilities: group IO, alignment, and group quality.

The three of those are now three modules. This stays as the name they are reached by, for
two reasons rather than inertia.

Forty-odd call sites import from here, and the names they ask for are the ones a caller
thinks of together: loading a dyad, aligning it, and finding out what it is worth are one
task even though they are three files. Splitting the file did not split the job.

The second reason is a test seam. ``pipeline.wtc_null`` imports ``compute_wtc_pseudo`` inside
its call, and the wiring tests patch it **on this module**. A lookup made at call time sees
the patch; a name bound at import time does not. Re-exporting the synchrony set here is what
keeps that working, and it is why those imports are lazy over there.

Layering, which the split made explicit and which nothing here should undo::

    group_io        reads a group off disk; depends on no other group module
    alignment       puts members on one clock; depends on no other group module
    group_quality   reduces them to verdicts; reads group_io, and nothing reads it
"""

from __future__ import annotations

from fnirs_pipe.pipeline.alignment import (  # noqa: F401  re-exported
    _stamp_alignment,
    align_like,
    align_recordings,
    alignment_params,
    crop_aligned_window,
    normalize_raws,
    resolve_analysis_window,
    trim_to_shortest,
)
from fnirs_pipe.pipeline.group_io import (  # noqa: F401  re-exported
    GroupEntry,
    _for_task,
    _hyper_sidecar,
    _member_sqm_files,
    _raw_to_haemo,
    load_group_haemo,
    load_group_raw_bids,
    load_group_stage,
    parse_group_csv,
    unfiltered_stage_note,
    warn_outside_passband,
)
from fnirs_pipe.pipeline.group_quality import (  # noqa: F401  re-exported
    _bad_from_csv,
    _bad_from_sidecar,
    _pairwise,
    _sci_from_sidecar,
    _screen_cutoffs,
    _screen_windows,
    apply_group_bads,
    compute_group_sqm_raw,
    load_group_sqm,
    resolve_group_bands,
    write_group_bads,
)
from fnirs_pipe.pipeline.synchrony import (  # noqa: F401  re-exported
    WTCResult,
    compute_pairwise_coherence,
    compute_wtc,
    compute_wtc_pseudo,
    roi_maps_from_channels,
    roi_mean_of_channels,
    window_result,
    wtc_band_mean,
)
