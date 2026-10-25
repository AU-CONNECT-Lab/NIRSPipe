"""A channel's cortical region: the annotation's region at the pial vertex nearest its position.

Expected regions are read with MNE's own annotation reader, not the one the package uses, at
pial vertices whose coordinates are fed back in as channel positions.
"""

import mne
import numpy as np

from nirspipe.io.head_template import (FSAVERAGE, cortex_atlases, nearest_cortex_regions,
                                       subjects_dir, template_dir)


def _labels(parc, hemi):
    return mne.read_labels_from_annot(FSAVERAGE, parc=parc, hemi=hemi,
                                      subjects_dir=subjects_dir(), verbose="error")


def _region_of(parc, hemi, vertex):
    return next((label.name for label in _labels(parc, hemi) if vertex in label.vertices), None)


def _pial(hemi, vertex):
    return mne.read_surface(template_dir(FSAVERAGE) / "surf" / f"{hemi}.pial")[0][vertex]


def _vertex_in(hemi, region):
    label = next(lab for lab in _labels("aparc", hemi) if lab.name == f"{region}-{hemi}")
    return int(label.vertices[len(label.vertices) // 2])


def test_a_point_on_a_pial_vertex_takes_that_vertexs_regions_in_either_hemisphere():
    vertices = [("lh", _vertex_in("lh", "precentral")), ("rh", _vertex_in("rh", "superiorfrontal"))]
    points = [_pial(hemi, v) for hemi, v in vertices] + [np.full(3, np.nan)]
    assert nearest_cortex_regions(FSAVERAGE, np.array(points)) == {
        "aparc": ["precentral-lh", "superiorfrontal-rh", None],
        "brodmann": [_region_of("PALS_B12_Brodmann", hemi, v) for hemi, v in vertices] + [None],
    }


def test_a_vertex_in_no_aparc_region_is_left_blank():
    named = set().union(*(set(label.vertices) for label in _labels("aparc", "lh")))
    vertex = next(v for v in range(len(named) + 1) if v not in named)
    assert nearest_cortex_regions(FSAVERAGE, _pial("lh", vertex)[None])["aparc"] == [None]


def test_brodmann_is_labelled_on_fsaverage_alone():
    assert cortex_atlases(FSAVERAGE) == ["aparc", "PALS_B12_Brodmann"]
    assert cortex_atlases("ANTS6p0Months3T") == ["aparc"]
    # nothing placed reads no template file, so an unfetched infant template is not fetched
    assert nearest_cortex_regions("ANTS6p0Months3T", np.full((2, 3), np.nan)) == {
        "aparc": [None, None]}
