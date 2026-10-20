"""
gdop.py
Closed-form 2D GDOP - reference-invariant (centroid-of-unit-vectors
form), fully vectorized across sample points.

CORRECTNESS NOTE: an earlier version formed H by differencing active
unit vectors against one arbitrary reference anchor, which is
REFERENCE-DEPENDENT (implicitly assumes independent noise on measurements
that actually share correlated noise via the common reference). The
correct, reference-invariant form is:
    GDOP = sqrt(tr[Sigma^-1]),  Sigma = sum_i (u_i - u_bar)(u_i - u_bar)^T
i.e. the covariance of active anchors' unit direction vectors around
their own centroid - verified reference-invariant to machine precision.
"""

import numpy as np


def gdop_2d(anchor_xy, point_xy):
    anchor_xy = np.asarray(anchor_xy, dtype=float)
    n = len(anchor_xy)
    if n < 2:
        return np.inf
    diffs = anchor_xy - point_xy
    dists = np.linalg.norm(diffs, axis=1)
    dists[dists == 0] = 1e-6
    u = diffs / dists[:, None]
    u_bar = u.mean(axis=0)
    centered = u - u_bar
    a = np.sum(centered[:, 0] ** 2)
    b = np.sum(centered[:, 0] * centered[:, 1])
    d = np.sum(centered[:, 1] ** 2)
    det = a * d - b * b
    if det <= 1e-9:
        return np.inf
    return np.sqrt((a + d) / det)


def gdop_all_points(anchor_positions, sample_points, active_mask):
    N, K = active_mask.shape
    diffs = anchor_positions[:, None, :] - sample_points[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    dists[dists == 0] = 1e-6
    u = diffs / dists[:, :, None]

    mask_f = active_mask.astype(float)
    n_active = mask_f.sum(axis=0)
    safe_n = np.where(n_active > 0, n_active, 1.0)
    u_masked = u * mask_f[:, :, None]
    u_bar = u_masked.sum(axis=0) / safe_n[:, None]

    centered = (u - u_bar[None, :, :]) * mask_f[:, :, None]
    a = np.einsum('nk,nk->k', centered[:, :, 0], centered[:, :, 0])
    b = np.einsum('nk,nk->k', centered[:, :, 0], centered[:, :, 1])
    d = np.einsum('nk,nk->k', centered[:, :, 1], centered[:, :, 1])
    det = a * d - b * b

    gdop = np.full(K, np.inf)
    valid = (n_active >= 2) & (det > 1e-9)
    gdop[valid] = np.sqrt((a[valid] + d[valid]) / det[valid])
    return gdop


if __name__ == "__main__":
    point = np.array([1.0, 2.0])
    anchors5 = np.array([[10, 3], [2, 15], [-8, 4], [5, -9], [12, 7]])
    print("Reference-invariance check:")
    for i in range(len(anchors5)):
        reordered = np.roll(anchors5, -i, axis=0)
        print(f"  order {i}: GDOP =", gdop_2d(reordered, point))

    rng = np.random.default_rng(0)
    anchors = rng.uniform(0, 50, size=(20, 2))
    points = rng.uniform(0, 30, size=(15, 2))
    active_mask = rng.random((20, 15)) < 0.4
    vec = gdop_all_points(anchors, points, active_mask)
    loop = np.array([gdop_2d(anchors[active_mask[:, k]], points[k]) if active_mask[:, k].sum() >= 2 else np.inf
                      for k in range(15)])
    print("Vectorized matches loop-based:", np.allclose(vec, loop, equal_nan=True))
