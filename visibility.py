"""
visibility.py
LOS/NLOS determination between candidate anchors and vehicle sample
points, with mixed-height obstacles (full-height columns, partial-
height parked cars, height-aware for the latter).
"""

import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


def _crossing_t_range(seg, seg_len, footprint):
    inter = seg.intersection(footprint)
    if inter.is_empty:
        return None
    coords = []
    if inter.geom_type == "Point":
        coords = [(inter.x, inter.y)]
    elif inter.geom_type == "LineString":
        coords = list(inter.coords)
    elif inter.geom_type in ("MultiLineString", "GeometryCollection"):
        for geom in getattr(inter, "geoms", []):
            if hasattr(geom, "coords"):
                coords.extend(list(geom.coords))
    if not coords:
        return None
    a = Point(seg.coords[0])
    ts = [a.distance(Point(c)) / seg_len for c in coords]
    return min(ts), max(ts)


def build_visibility_matrix(anchor_points, sample_points, columns, cars, an_height, tag_height):
    N, K = len(anchor_points), len(sample_points)
    vis_matrix = np.ones((N, K), dtype=bool)
    column_tree = STRtree(columns)
    car_polys = [c["polygon"] for c in cars]
    car_heights = [c["height"] for c in cars]
    car_tree = STRtree(car_polys) if car_polys else None
    dh = tag_height - an_height

    for i, a in enumerate(anchor_points):
        for k, p in enumerate(sample_points):
            seg = LineString([tuple(a), tuple(p)])
            seg_len = seg.length
            if seg_len == 0:
                continue
            blocked = False
            for idx in column_tree.query(seg):
                if seg.intersects(columns[idx]):
                    blocked = True
                    break
            if not blocked and car_tree is not None:
                for idx in car_tree.query(seg):
                    footprint = car_polys[idx]
                    if not seg.intersects(footprint):
                        continue
                    t_range = _crossing_t_range(seg, seg_len, footprint)
                    if t_range is None:
                        continue
                    _, t_exit = t_range
                    min_h = an_height + t_exit * dh
                    if min_h < car_heights[idx]:
                        blocked = True
                        break
            vis_matrix[i, k] = not blocked
    return vis_matrix


def los_count_per_sample(vis_matrix):
    return vis_matrix.sum(axis=0)


if __name__ == "__main__":
    import time
    from garage_model import build_scenario, GARAGE_1, GARAGE_2
    for cfg in (GARAGE_1, GARAGE_2):
        scenario = build_scenario(cfg)
        t0 = time.time()
        vis = build_visibility_matrix(scenario["candidate_AN_points"], scenario["vehicle_sample_points"],
                                       scenario["columns"], scenario["parked_cars"],
                                       cfg.an_height, cfg.tag_height)
        los = los_count_per_sample(vis)
        print(f"{cfg.name}: {time.time()-t0:.2f}s, mean LOS={los.mean():.1f}, min LOS={los.min()}")
