"""--events-path feeds the GLM design only; the run report's epoch figures and trial QC keep the recording's markers."""

from tests.figure_accuracy.conftest import run_capturing
from tests._fingerprint import EVENT_DURATION, EVENT_ONSETS


def test_the_epoch_figures_and_trial_qc_keep_the_recording_s_markers(tmp_path):
    events = tmp_path / "events.tsv"
    events.write_text("onset\tduration\ttrial_type\n" + "".join(
        f"{o:g}\t{EVENT_DURATION:g}\tFinger\n" for o in EVENT_ONSETS), encoding="utf-8")
    done = run_capturing(tmp_path, ["--mode", "glm", "--drift-high-pass", "0.008",
                                    "--short-channel", "mean", "--events-path", str(events)],
                         ("build_epoch_preview_figure", "design_matrix_static_figure",
                          "trial_quality_heatmap"))
    assert "Finger" in done["captured"]["design_matrix_static_figure"][0][0].columns
    # the fingerprint recording marks its trials with the task's name
    drawn = done["captured"]["build_epoch_preview_figure"][0][0]
    assert {d for d in drawn.annotations.description if not d.startswith("BAD_")} == {"tapping"}
    labels = done["captured"]["trial_quality_heatmap"][0][0]
    assert len(labels) == len(EVENT_ONSETS)
    assert all(label.endswith("_tapping") for label in labels)
