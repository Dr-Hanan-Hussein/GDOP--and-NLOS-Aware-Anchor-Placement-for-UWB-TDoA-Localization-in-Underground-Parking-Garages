"""
tdoa_solver.py
TDoA positioning solver and Monte Carlo helpers (Section 3.2).
  - Chan & Ho (1994) closed-form initialisation of the nonlinear least-squares refinement (bounded trust-region-reflective),
    replacing a naive centroid start.
  - Distance-dependent range noise, sigma(d) = floor + coeff * d, applied to each anchor's own range (Eqs. 16-17).
"""

import numpy as np
from scipy.optimize import least_squares


def ranges(anchor_xy, p_xy, dz):
    dx = p_xy[0] - anchor_xy[:, 0]
    dy = p_xy[1] - anchor_xy[:, 1]
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def simulate_measurements(anchor_xy, p_true, sigma_floor, sigma_coeff, rng, dz):
    """
    Distance-dependent noise model: each anchor's OWN
    range measurement gets noise ~ N(0, sigma(d_i)^2) where d_i is that
    anchor's TRUE distance to the point - not a single flat sigma
    applied directly to the range difference. This is more physically
    correct (each anchor's timing jitter is independent and depends on
    ITS OWN link quality/SNR, which depends on ITS OWN range) and is
    what the range-difference noise structure of the TDoA model is
    actually meant to represent.
    """
    r = ranges(anchor_xy, p_true, dz)
    sigma_per_anchor = sigma_floor + sigma_coeff * r
    range_noise = rng.normal(0, sigma_per_anchor)
    r_noisy = r + range_noise
    return r_noisy[1:] - r_noisy[0]  # measured TDoA relative to anchor 0


def chan_ho_initial_estimate(anchor_xy, D_measured):
    """
    Chan & Ho (1994) closed-form TDoA position estimate, used as the
    initial guess for the nonlinear least-squares refinement. Linearizes the TDoA equations by introducing r_0 (range to the
    reference anchor) as an auxiliary unknown, then solves the resulting
    overdetermined LINEAR system for (x, y, r_0) via ordinary least
    squares - no iteration needed.

    Derivation: for anchor i (i=1..n-1) vs reference anchor 0,
        r_i^2 - r_0^2 = (x_i^2+y_i^2-x_0^2-y_0^2) - 2(x_i-x_0)x - 2(y_i-y_0)y
    and since r_i = D_i0 + r_0:
        r_i^2 - r_0^2 = D_i0*(D_i0 + 2*r_0)
    Equating and rearranging into A*[x,y,r_0]^T = b gives one linear
    equation per anchor pair - solved by least squares when n>=4 anchors
    give an overdetermined (n-1 >= 3 equation) system.

    Returns: (x, y) estimate. Falls back to the anchor centroid if the
    linear system is singular/ill-conditioned (rare, e.g. exactly
    collinear anchors).
    """
    x0, y0 = anchor_xy[0]
    xi, yi = anchor_xy[1:, 0], anchor_xy[1:, 1]
    Di0 = D_measured

    A = np.column_stack([
        2 * (xi - x0),
        2 * (yi - y0),
        2 * Di0,
    ])
    b = (xi ** 2 + yi ** 2 - x0 ** 2 - y0 ** 2) - Di0 ** 2

    try:
        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        if rank < 3 or not np.all(np.isfinite(sol)):
            raise np.linalg.LinAlgError("rank-deficient")
        return np.array([sol[0], sol[1]])
    except np.linalg.LinAlgError:
        return anchor_xy.mean(axis=0)  # fallback: naive centroid


def solve_position(anchor_xy, D_measured, dz, p_init=None, use_chan_ho=True, bounds=None):
    """
    Nonlinear least squares solve for position (Eq. 4). Uses the
    Trust-Region-Reflective method with physical bounds, not
    unconstrained Levenberg-Marquardt: diagnosed a real numerical
    divergence failure mode where LM, started from a perfectly
    reasonable Chan-Ho estimate (~5m off), took an unstable step and
    diverged to a position tens of km away. Since we always know the
    vehicle is physically within (or near) the garage, bounding the
    search space is standard, well-justified practice - not a hack -
    and eliminates this divergence mode entirely.
    """
    if p_init is None:
        p_init = chan_ho_initial_estimate(anchor_xy, D_measured) if use_chan_ho \
            else anchor_xy.mean(axis=0)

    if bounds is None:
        margin = 20.0  # m, generous slack beyond the anchor extent
        lo = anchor_xy.min(axis=0) - margin
        hi = anchor_xy.max(axis=0) + margin
        bounds = (lo, hi)
        p_init = np.clip(p_init, lo, hi)  # TRF requires init strictly within bounds

    def residuals(p_xy):
        r = ranges(anchor_xy, p_xy, dz)
        return D_measured - (r[1:] - r[0])

    result = least_squares(residuals, p_init, method='trf', bounds=bounds)
    return result.x, result.success


def monte_carlo_trial(anchor_xy, p_true, sigma_floor, sigma_coeff, n_trials, rng, dz, use_chan_ho=True):
    errors = np.empty(n_trials)
    n_success = 0
    for t in range(n_trials):
        D = simulate_measurements(anchor_xy, p_true, sigma_floor, sigma_coeff, rng, dz)
        p_hat, ok = solve_position(anchor_xy, D, dz, use_chan_ho=use_chan_ho)
        errors[t] = np.linalg.norm(p_hat - p_true)
        n_success += ok
    return errors, n_success / n_trials


if __name__ == "__main__":
    from garage_model import GARAGE_1, GARAGE_2
    from gdop import gdop_2d

    rng = np.random.default_rng(0)
    anchors = np.array([[0, 0], [20, 0], [10, 20], [0, 20], [20, 20]], dtype=float)
    p_true = np.array([9.0, 8.0])

    print("Sanity check: Chan-Ho initial estimate vs naive centroid")
    for cfg in (GARAGE_1, GARAGE_2):
        dz = cfg.tag_height - cfg.an_height
        D = simulate_measurements(anchors, p_true, cfg.sigma_tdoa_floor, cfg.sigma_tdoa_coeff, rng, dz)
        chan_est = chan_ho_initial_estimate(anchors, D)
        centroid_est = anchors.mean(axis=0)
        print(f"  {cfg.name}:")
        print(f"    Chan-Ho init error:  {np.linalg.norm(chan_est-p_true)*100:.1f}cm")
        print(f"    Centroid init error: {np.linalg.norm(centroid_est-p_true)*100:.1f}cm")

        errors, sr = monte_carlo_trial(anchors, p_true, cfg.sigma_tdoa_floor, cfg.sigma_tdoa_coeff, 300, rng, dz)
        rmse = np.sqrt(np.mean(errors ** 2))
        g = gdop_2d(anchors, p_true)
        avg_dist = np.linalg.norm(anchors - p_true, axis=1).mean()
        effective_sigma = cfg.sigma_tdoa_floor + cfg.sigma_tdoa_coeff * avg_dist
        print(f"    Final RMSE={rmse*100:.1f}cm, GDOP={g:.3f}, effective_sigma={effective_sigma*100:.1f}cm, "
              f"predicted={g*effective_sigma*100:.1f}cm\n")
