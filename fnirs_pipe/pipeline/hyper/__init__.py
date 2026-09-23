"""The hyperscanning half of the pipeline, and the one name its group modules are reached by.

Keep the re-exports. The wiring tests patch ``compute_wtc_phase_null`` on *this* module and
``wtc_null`` looks it up at call time, so dropping a name here breaks a seam, not an import.

Layering, which nothing should undo::

    group_io        reads a group off disk; depends on no other group module
    alignment       puts members on one clock; depends on no other group module
    group_quality   reduces them to verdicts; reads group_io, and nothing reads it
"""

from __future__ import annotations

from fnirs_pipe.pipeline.hyper.alignment import (  # noqa: F401  re-exported
    _stamp_alignment,
    align_like,
    align_recordings,
    alignment_params,
    crop_aligned_window,
    normalize_raws,
    resolve_analysis_window,
    trim_to_shortest,
    write_aligned_member,
)
from fnirs_pipe.pipeline.hyper.group_io import (  # noqa: F401  re-exported
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
from fnirs_pipe.pipeline.hyper.group_quality import (  # noqa: F401  re-exported
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
from fnirs_pipe.pipeline.hyper.coherence import (  # noqa: F401  re-exported
    compute_pairwise_coherence,
)
from fnirs_pipe.pipeline.hyper.roi import (  # noqa: F401  re-exported
    roi_maps_from_channels,
    roi_mean_of_channels,
)
from fnirs_pipe.pipeline.hyper.surrogate import (  # noqa: F401  re-exported
    compute_wtc_pair_null,
    compute_wtc_phase_null,
)
from fnirs_pipe.pipeline.hyper.wtc import (  # noqa: F401  re-exported
    WTCResult,
    compute_wtc,
    window_result,
    wtc_band_mean,
)
