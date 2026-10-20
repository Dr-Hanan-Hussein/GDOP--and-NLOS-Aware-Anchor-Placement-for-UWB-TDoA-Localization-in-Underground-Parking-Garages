"""
garage_model.py

Parametric underground garage geometry model. Two named configurations:
  GARAGE_1: original synthetic reference garage (50m x 30m)
  GARAGE_2: smaller, denser garage informed by real published parameters
            from Jiang et al. 2022 ("A Practical and Economical
            Ultra-wideband Base Station Placement Approach for Indoor
            Autonomous Driving Systems", J. Adv. Transport., 2022:3815306),
            case study: real underground parking lot at Tongji University,
            Shanghai (anchor height 2.6m, ranging noise 5cm - both real
            published values). Overall footprint is a size-representative
            reconstruction, not exact geometric replication of the real
            (irregular, multi-zone) site.
"""

from dataclasses import dataclass
import numpy as np
from shapely.geometry import Point, box


@dataclass(frozen=True)
class GarageConfig:
    name: str
    width: float
    length: float
    column_grid_spacing: float
    column_size: float
    ceiling_height: float
    an_height: float
    tag_height: float
    wall_margin: float
    stall_width: float
    stall_depth: float
    aisle_width: float
    car_width: float
    car_length: float
    car_height: float
    occupancy_rate: float
    random_seed: int
    sigma_tdoa_floor: float       # m, base ranging noise at d->0 (electronics/clock jitter)
    sigma_tdoa_coeff: float       # m per m, growth rate with distance (multipath/SNR falloff)


GARAGE_1 = GarageConfig(
    name="Garage 1 (reference, synthetic)",
    width=50.0, length=30.0, column_grid_spacing=8.0, column_size=0.4,
    ceiling_height=2.8, an_height=2.0, tag_height=0.5, wall_margin=1.0,
    stall_width=2.5, stall_depth=5.3, aisle_width=6.0,
    car_width=1.8, car_length=4.5, car_height=1.5, occupancy_rate=0.80,
    random_seed=42, sigma_tdoa_floor=0.05, sigma_tdoa_coeff=0.0025,
    # Distance-dependent TDoA noise model: sigma(d) = sigma_tdoa_floor + sigma_tdoa_coeff * d.
    # The growth of UWB range-error deviation with distance follows Bellusci et al. (2009); the values are
    # in line with vendor specifications of DW1000-class hardware (< 10 cm). The floor (5 cm) stands for
    # electronics/clock jitter; the coefficient is set so that sigma(20 m) ~= 10 cm, the typical LOS
    # anchor distance in this garage (Table 2 and Section 4.2 of the paper).
)

GARAGE_2 = GarageConfig(
    name="Garage 2 (Tongji University-inspired, smaller/denser)",
    width=30.0, length=20.0, column_grid_spacing=6.0, column_size=0.4,
    ceiling_height=3.0, an_height=2.6, tag_height=0.5, wall_margin=1.0,
    stall_width=2.5, stall_depth=5.3, aisle_width=5.5,
    car_width=1.8, car_length=4.5, car_height=1.5, occupancy_rate=0.85,
    random_seed=100, sigma_tdoa_floor=0.03, sigma_tdoa_coeff=0.002,
    # Calibrated so sigma(10m)~=5cm, matching Jiang et al. 2022's real
    # measured value at Garage 2's smaller typical distance scale -
    # same physical model as Garage 1, different (better, real
    # measured) hardware-level calibration.
)


def build_garage_boundary(cfg):
    return box(0, 0, cfg.width, cfg.length)


def build_columns(cfg):
    columns = []
    x_positions = np.arange(cfg.wall_margin + cfg.column_grid_spacing / 2,
                             cfg.width - cfg.wall_margin, cfg.column_grid_spacing)
    y_positions = np.arange(cfg.wall_margin + cfg.column_grid_spacing / 2,
                             cfg.length - cfg.wall_margin, cfg.column_grid_spacing)
    half = cfg.column_size / 2
    for cx in x_positions:
        for cy in y_positions:
            columns.append(box(cx - half, cy - half, cx + half, cy + half))
    return columns


def build_stall_row_bands(cfg):
    y = cfg.wall_margin
    stall_bands, aisle_bands = [], []
    while y + cfg.stall_depth <= cfg.length - cfg.wall_margin + 1e-6:
        stall_bands.append((y, y + cfg.stall_depth))
        y += cfg.stall_depth
        if y + cfg.aisle_width <= cfg.length - cfg.wall_margin + 1e-6:
            aisle_bands.append((y, y + cfg.aisle_width))
            y += cfg.aisle_width
        else:
            break
    return stall_bands, aisle_bands


def build_parked_cars(cfg, columns):
    rng = np.random.default_rng(cfg.random_seed)
    stall_bands, _ = build_stall_row_bands(cfg)
    cars = []
    x_positions = np.arange(cfg.wall_margin, cfg.width - cfg.wall_margin, cfg.stall_width)
    for (y_start, y_end) in stall_bands:
        y_center = (y_start + y_end) / 2
        for x_start in x_positions:
            x_center = x_start + cfg.stall_width / 2
            stall_center = Point(x_center, y_center)
            if any(stall_center.distance(c) < 0.5 for c in columns):
                continue
            if rng.random() < cfg.occupancy_rate:
                car_poly = box(x_center - cfg.car_width / 2, y_center - cfg.car_length / 2,
                                x_center + cfg.car_width / 2, y_center + cfg.car_length / 2)
                cars.append({"polygon": car_poly, "height": cfg.car_height})
    return cars


def build_candidate_AN_points(cfg, columns):
    candidates = []
    x_nodes = np.arange(cfg.wall_margin, cfg.width - cfg.wall_margin + 1e-6, cfg.column_grid_spacing / 2)
    y_nodes = np.arange(cfg.wall_margin, cfg.length - cfg.wall_margin + 1e-6, cfg.column_grid_spacing / 2)
    for x in x_nodes:
        for y in y_nodes:
            pt = Point(x, y)
            if not any(pt.distance(c) < 0.3 for c in columns):
                candidates.append((x, y))
    return np.array(candidates)


def build_vehicle_sample_grid(cfg, columns, spacing=1.5):
    _, aisle_bands = build_stall_row_bands(cfg)
    x_vals = np.arange(0.5, cfg.width, spacing)
    points = []
    for (y_start, y_end) in aisle_bands:
        y_vals = np.arange(y_start + 0.5, y_end, spacing)
        for x in x_vals:
            for y in y_vals:
                pt = Point(x, y)
                if not any(pt.distance(c) < 0.3 for c in columns):
                    points.append((x, y))
    return np.array(points)


def build_scenario(cfg=GARAGE_1):
    boundary = build_garage_boundary(cfg)
    columns = build_columns(cfg)
    cars = build_parked_cars(cfg, columns)
    candidates = build_candidate_AN_points(cfg, columns)
    samples = build_vehicle_sample_grid(cfg, columns)
    return {"config": cfg, "boundary": boundary, "columns": columns,
            "parked_cars": cars, "candidate_AN_points": candidates,
            "vehicle_sample_points": samples}


if __name__ == "__main__":
    for cfg in (GARAGE_1, GARAGE_2):
        s = build_scenario(cfg)
        print(f"{cfg.name}: {cfg.width}m x {cfg.length}m ({cfg.width*cfg.length:.0f} m^2)")
        print(f"  Columns: {len(s['columns'])}, Cars: {len(s['parked_cars'])}, "
              f"Candidates: {len(s['candidate_AN_points'])}, Samples: {len(s['vehicle_sample_points'])}")
