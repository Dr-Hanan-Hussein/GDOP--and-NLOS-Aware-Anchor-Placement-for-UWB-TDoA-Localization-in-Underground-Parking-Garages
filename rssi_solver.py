"""
rssi_solver.py
RSSI-based positioning for the WiFi/BLE benchmark (Section 5.4, Table 8; log-distance model of Eq. 19).
Independent absolute range estimates per AP (via path-loss inversion),
solved via standard multilateration (not TDoA range-differences).
"""

import numpy as np
from scipy.optimize import least_squares


def simulate_rssi_ranges(ap_xy, p_true, A, n, sigma_shadow_db, rng, nlos_mask=None, nlos_penalty_db=0.0):
    true_d = np.maximum(np.linalg.norm(ap_xy - p_true, axis=1), 0.1)
    shadow_noise = rng.normal(0, sigma_shadow_db, size=len(ap_xy))
    rssi = -10 * n * np.log10(true_d) + A + shadow_noise
    if nlos_mask is not None:
        rssi = rssi - nlos_mask.astype(float) * nlos_penalty_db
    return 10 ** ((A - rssi) / (10 * n))


def solve_position_rssi(ap_xy, d_hat, p_init=None):
    if p_init is None:
        p_init = ap_xy.mean(axis=0)

    def residuals(p_xy):
        return d_hat - np.linalg.norm(ap_xy - p_xy, axis=1)

    result = least_squares(residuals, p_init, method='lm')
    return result.x, result.success


def monte_carlo_trial_rssi(ap_xy, p_true, A, n, sigma_shadow_db, n_trials, rng,
                            nlos_mask=None, nlos_penalty_db=0.0):
    errors = np.empty(n_trials)
    n_success = 0
    for t in range(n_trials):
        d_hat = simulate_rssi_ranges(ap_xy, p_true, A, n, sigma_shadow_db, rng, nlos_mask, nlos_penalty_db)
        p_hat, ok = solve_position_rssi(ap_xy, d_hat)
        errors[t] = np.linalg.norm(p_hat - p_true)
        n_success += ok
    return errors, n_success / n_trials


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    aps = np.array([[0, 0], [40, 0], [20, 35], [0, 35]], dtype=float)
    p_true = np.array([15.0, 12.0])
    errors, sr = monte_carlo_trial_rssi(aps, p_true, A=-40, n=3.5, sigma_shadow_db=6.0, n_trials=300, rng=rng)
    print(f"WiFi calibration check: RMSE={np.sqrt(np.mean(errors**2)):.2f}m (target 10-15m)")
