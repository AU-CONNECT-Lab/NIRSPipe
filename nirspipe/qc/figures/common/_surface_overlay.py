"""Per-vertex colours for an activation map on a cortical surface, drawn as ``stc.plot`` does.

A binary curvature layer in grey sits under a diverging data layer that is transparent near
zero; the two are composited into one RGBA per vertex, ready to hand to a mesh.
"""

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from scipy import sparse

_N_COLORS = 256


def _unit_rows(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norm > 0, norm, 1.0)


def vertex_normals(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normal per vertex: the normalised sum of the unit normals of the faces around it."""
    r1, r2, r3 = (coords[faces[:, i]] for i in range(3))
    face_nn = _unit_rows(np.cross(r2 - r1, r3 - r1))
    nn = np.zeros((len(coords), 3))
    for corner in range(3):
        for axis in range(3):
            nn[:, axis] += np.bincount(faces[:, corner], weights=face_nn[:, axis],
                                       minlength=len(coords))
    return _unit_rows(nn)


def smoothing_matrix(faces: np.ndarray, n_vertices: int, vertices_from: np.ndarray,
                     steps: int) -> sparse.csr_array:
    """Sparse ``(n_vertices, len(vertices_from))`` map spreading values from a vertex subset.

    Each step replaces every vertex reached so far, and every neighbour of one, by the mean
    of itself and its reached neighbours. One step from a single vertex of a triangle fan
    gives that vertex and its ring equal weight; later steps widen and flatten the patch.
    """
    a, b, c = faces.T
    adj = sparse.coo_array((np.ones(3 * len(faces)), (np.r_[a, b, c], np.r_[b, c, a])),
                           shape=(n_vertices, n_vertices)).tocsr()
    adj = adj + adj.T
    adj.data[adj.data == 2] = 1                  # an edge shared by two faces counts once
    adj = adj + sparse.eye_array(n_vertices, format="csr")

    data = sparse.eye_array(len(vertices_from), format="csr")
    reached = np.asarray(vertices_from)
    seen = np.zeros(n_vertices)
    for step in range(steps):
        if step:
            data = data[reached]
        data = adj[:, reached] @ data
        seen[reached] = 1
        row_sum = adj @ seen                     # reached neighbours, self included
        reached = np.flatnonzero(row_sum)
        data.data /= np.where(row_sum, row_sum, 1).repeat(np.diff(data.indptr))
    return data


def _stretch(table: np.ndarray, lo: float, mid: float, hi: float) -> np.ndarray:
    """Resample a sequential table so its midpoint colour lands on ``mid`` within [lo, hi]."""
    n = table.shape[0]
    mid_idx = 0 if hi == lo else int(np.clip(np.round(n * (mid - lo) / (hi - lo)) - 1,
                                             0, n - 2))
    n_left, n_right = mid_idx + 1, n - mid_idx - 1
    out = table.copy()
    for ch in range(4):
        out[:n_left, ch] = np.interp(np.linspace(0, n // 2 - 1, n_left), np.arange(n),
                                     table[:, ch])
        out[n_left:, ch] = np.interp(np.linspace(n - 1, n // 2, n_right)[::-1], np.arange(n),
                                     table[:, ch])
    return out


def _centre_fill(cols: np.ndarray, n_fill: int) -> np.ndarray:
    """Colours for the hidden band around zero, taken from the table's own centre."""
    steps = np.linalg.norm(np.diff(cols[:, :3].astype(float), axis=0), axis=1)
    jump = np.flatnonzero(steps[1:-1] > steps[[0, -1]].mean() * 3)
    if jump.size:
        i = jump[0] + 1
        return np.r_[np.tile(cols[i], (n_fill // 2, 1)),
                     np.tile(cols[i + 1], (n_fill - n_fill // 2, 1))]
    return np.tile(cols[cols.shape[0] // 2], (n_fill, 1))


def diverging_lut(fmin: float, fmid: float, fmax: float, cmap: str = "RdBu_r") -> np.ndarray:
    """``(256, 4)`` uint8 RGBA table spanning [-fmax, fmax], transparent within ±fmin.

    Opacity rises linearly from 0 at ±fmin to full at ±fmid, so small values fade into the
    cortex instead of tinting it::

        diverging_lut(0, 0.5, 1)[128, 3] -> 0      # zero is fully transparent
        diverging_lut(0, 0.5, 1)[0, 3]   -> 255    # -fmax is opaque
    """
    if not fmin <= fmid <= fmax or fmin == fmax:
        raise ValueError(f"need fmin <= fmid <= fmax with fmin < fmax, "
                         f"got {fmin}, {fmid}, {fmax}")
    half, quarter = _N_COLORS // 2, _N_COLORS // 4
    table = np.round(colormaps[cmap](np.linspace(0, 1, _N_COLORS)) * 255.0).astype(np.int64)
    table[:, 3] = np.round(np.r_[np.full(quarter, 255.0), np.linspace(0, 255, quarter)[::-1],
                                 np.linspace(0, 255, quarter), np.full(quarter, 255.0)])

    # each half stretched so its fmid lands where asked; the band inside ±fmin is filled
    n_fill = int(round(fmin * half / (fmax - fmin))) * 2
    table = np.r_[_stretch(table[:half], -fmax, -fmid, -fmin),
                  _centre_fill(table[half - 3:half + 3], n_fill),
                  _stretch(table[half:][::-1], -fmax, -fmid, -fmin)[::-1]]
    if len(table) != _N_COLORS:
        x = np.linspace(1, len(table), _N_COLORS)
        table = np.column_stack([np.interp(x, np.arange(1, len(table) + 1), table[:, ch])
                                 for ch in range(4)])
    return np.round(table.astype(np.float64) / 255.0 * 255).astype(np.uint8)


def _over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """``top`` composited over ``bottom``, both straight (not premultiplied) RGBA in [0, 1]."""
    top_a = top[:, 3:]
    bottom_w = bottom[:, 3:] * (1 - top_a)
    out = top.copy()
    out[:, :3] = top[:, :3] * top_a + bottom[:, :3] * bottom_w
    out[:, 3:] = top_a + bottom_w
    seen = out[:, 3] != 0
    out[seen, :3] /= out[seen, 3:]
    out[~seen, :3] = 0
    return np.clip(out, 0, 1, out=out)


def activation_rgba(curvature: np.ndarray, values: np.ndarray,
                    fmin: float, fmid: float, fmax: float) -> np.ndarray:
    """Per-vertex RGBA: sulci and gyri in two greys, with ``values`` over them."""
    # gyri (curvature <= 0) at a third of the grey ramp, sulci at two thirds
    cortex = colormaps["Greys"]((np.asarray(curvature > 0, float) + 1) / 3)
    lut = ListedColormap(diverging_lut(fmin, fmid, fmax) / 255.0)
    data = lut((values + fmax) / (2 * fmax))
    return _over(cortex, data)
