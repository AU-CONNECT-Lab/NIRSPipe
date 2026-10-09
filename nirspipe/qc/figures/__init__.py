from fnirs_pipe.qc.figures.subject.raw_figures import (
    build_ts_figure,
    build_channel_figure,
    build_layout_figure,
    build_psd_mean_figure,
        build_epoch_preview_figure,
    build_trial_image_figure,
    build_roi_trial_image_figure,
    build_trial_image_by_condition,
    build_roi_trial_image_by_condition,
    build_evoked_topo_figure,
    build_trigger_timeline_single,
    condition_colors,
    sci_color,
    psd_layout,
)
from fnirs_pipe.qc.figures.subject.sci_psp_panel import (
    build_sci_psp_figure,
    binary_heatmap_figure,
    lollipop_scores_figure,
    channel_quality_heatmap,
    condition_quality_heatmap,
    trial_quality_heatmap,
)
from fnirs_pipe.qc.figures.subject.brain_views import quality_brain_views
from fnirs_pipe.qc.figures.common.motion_panel import (
    carpet_gvtd_figure, bad_segment_zoom_figure, build_motion_detail_figure,
)
from fnirs_pipe.qc.figures.subject.carpet_compare import carpet_compare_figure
from fnirs_pipe.qc.figures.subject.psd_plot import psd_figure
from fnirs_pipe.qc.figures.subject.denoise_compare import (
    denoise_stage_panels, stage_metrics_figure, stage_slope_figure,
)
from fnirs_pipe.qc.figures.subject.glm_figures import (
    design_matrix_figure, design_matrix_static_figure,
    activation_brain_figure, activation_condition_figures, activation_panel,
    per_channel_hrf_figure, design_matrix_heatmap, glm_betas_figure,
)
from fnirs_pipe.qc.figures.subject.correlation_panel import fit_js as hbo_hbr_fit_js
from fnirs_pipe.qc.figures.subject.correlation_panel import hbo_hbr_correlation_figure
from fnirs_pipe.qc.figures.subject.optode_layout import optode_layout_static
from fnirs_pipe.qc.figures.common.topomap import evoked_channel_map_figure
from fnirs_pipe.qc.figures.subject.rest_figures import (
    alff_topo_figure,
    rest_channel_panel,
    fc_roi_matrix_figure,
    fc_seed_topo_figure,
)

__all__ = [
    "build_ts_figure",
    "build_channel_figure",
    "build_layout_figure",
    "build_psd_mean_figure",
    "build_sci_psp_figure",
    "build_epoch_preview_figure",
    "build_trial_image_figure",
    "build_roi_trial_image_figure",
    "build_trial_image_by_condition",
    "build_roi_trial_image_by_condition",
    "build_evoked_topo_figure",
    "build_trigger_timeline_single",
    "condition_colors",
    "sci_color",
    "psd_layout",
    "binary_heatmap_figure",
    "lollipop_scores_figure",
    "channel_quality_heatmap",
    "condition_quality_heatmap",
    "trial_quality_heatmap",
    "quality_brain_views",
    "carpet_gvtd_figure",
    "bad_segment_zoom_figure",
    "build_motion_detail_figure",
    "carpet_compare_figure",
    "psd_figure",
    "denoise_stage_panels",
    "stage_metrics_figure",
    "stage_slope_figure",
    "design_matrix_figure",
    "design_matrix_static_figure",
    "activation_brain_figure",
    "activation_condition_figures",
    "activation_panel",
    "per_channel_hrf_figure",
    "design_matrix_heatmap",
    "glm_betas_figure",
    "hbo_hbr_correlation_figure",
    "hbo_hbr_fit_js",
    "optode_layout_static",
    "evoked_channel_map_figure",
    "alff_topo_figure",
    "rest_channel_panel",
    "fc_roi_matrix_figure",
    "fc_seed_topo_figure",
]
