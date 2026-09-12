import math
import random
import itertools
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Original 5 points (lat, lon)
# ---------------------------------------------------------------
NAMES = ["Nichols", "Chapman", "Stride", "Eddowes", "Kelly"]
POINTS_LATLON = [
    (51.5188, -0.0669),  # Nichols
    (51.5192, -0.0722),  # Chapman
    (51.5133, -0.0625),  # Stride
    (51.5142, -0.0784),  # Eddowes
    (51.5177, -0.0715),  # Kelly
]

def latlon_to_xy(lat, lon, ref_lat):
    x = lon * 111320 * math.cos(math.radians(ref_lat))
    y = lat * 110540
    return x, y

ref_lat = sum(p[0] for p in POINTS_LATLON) / len(POINTS_LATLON)
POINTS_XY = [latlon_to_xy(lat, lon, ref_lat) for lat, lon in POINTS_LATLON]

lats = [p[0] for p in POINTS_LATLON]
lons = [p[1] for p in POINTS_LATLON]
LAT_MIN, LAT_MAX = min(lats), max(lats)
LON_MIN, LON_MAX = min(lons), max(lons)

# ---------------------------------------------------------------
# 2. Geometry & Strict Vesica Piscis Scoring Engine
# ---------------------------------------------------------------
def line_intersect_params(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    denom = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(denom) < 1e-9:
        return None
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denom
    s = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / denom
    return t, s

def angle_between(p1, p2, p3, p4):
    v1 = (p2[0]-p1[0], p2[1]-p1[1])
    v2 = (p4[0]-p3[0], p4[1]-p3[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = math.hypot(*v1); n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0
    cosang = max(-1, min(1, dot / (n1 * n2)))
    ang = math.degrees(math.acos(cosang))
    return min(ang, 180 - ang)

def vesica_piscis_score(p1, p2, p3, p4):
    """
    Strict Vesica Piscis Metric:
    1. Angle Score: Requires perpendicular axes (~90°).
    2. Centering Score: Requires intersection near the midpoints.
    3. Strict Side Equality: Cubed ratio of shortest to longest side to heavily 
       penalize asymmetrical quadrilaterals/kites.
    4. Vesica Aspect Ratio: Penalizes deviation from the ideal Vesica ratio (sqrt(3) ≈ 1.732).
    """
    params = line_intersect_params(p1, p2, p3, p4)
    if params is None:
        return 0.0
    t, s = params
    if not (0 <= t <= 1 and 0 <= s <= 1):
        return 0.0

    # 1. Perpendicularity & Centering
    angle = angle_between(p1, p2, p3, p4)
    angle_score = 1 - abs(angle - 90) / 90
    centering_score = max(0, 1 - (abs(t - 0.5) + abs(s - 0.5)))

    # Intersection Center
    cx = p1[0] + t * (p2[0] - p1[0])
    cy = p1[1] + t * (p2[1] - p1[1])

    # Sort 4 points around center
    pts = [p1, p2, p3, p4]
    pts_sorted = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    # 2. Strict 4-side perimeter equality (cubed to strongly reject distorted shapes)
    edges = [
        math.hypot(pts_sorted[i][0] - pts_sorted[(i + 1) % 4][0],
                   pts_sorted[i][1] - pts_sorted[(i + 1) % 4][1])
        for i in range(4)
    ]
    min_edge, max_edge = min(edges), max(edges)
    if max_edge == 0:
        return 0.0
    edge_equality = (min_edge / max_edge) ** 3  

    # 3. Vesica proportion score (ideal diagonal ratio = sqrt(3))
    d1 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    d2 = math.hypot(p4[0] - p3[0], p4[1] - p3[1])
    d_max, d_min = max(d1, d2), min(d1, d2)
    if d_min == 0:
        return 0.0

    ideal_ratio = math.sqrt(3)
    actual_ratio = d_max / d_min
    ratio_score = max(0, 1 - abs(actual_ratio - ideal_ratio) / ideal_ratio)

    return angle_score * centering_score * edge_equality * ratio_score

def best_vesica_in_4(pts):
    a, b, c, d = pts
    splits = [
        ((a, b), (c, d)),
        ((a, c), (b, d)),
        ((a, d), (b, c)),
    ]
    best_score, best_split = -1, None
    for (p1, p2), (p3, p4) in splits:
        sc = vesica_piscis_score(p1, p2, p3, p4)
        if sc > best_score:
            best_score, best_split = sc, ((p1, p2), (p3, p4))
    return best_score, best_split

def best_vesica_in_n(points_xy):
    best_score, best_combo, best_split = -1, None, None
    n = len(points_xy)
    for idx in itertools.combinations(range(n), 4):
        pts = [points_xy[i] for i in idx]
        sc, split = best_vesica_in_4(pts)
        if sc > best_score:
            best_score, best_combo, best_split = sc, idx, split
    return best_score, best_combo, best_split

def plot_vesica_shape(ax, p1, p2, p3, p4):
    """
    Renders both the straight symmetrical frame and the signature curved Vesica lens arcs.
    """
    params = line_intersect_params(p1, p2, p3, p4)
    if params is None:
        return
    t, _ = params
    cx = p1[0] + t * (p2[0] - p1[0])
    cy = p1[1] + t * (p2[1] - p1[1])

    pts = [p1, p2, p3, p4]
    pts_sorted = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    # 1. Straight outer frame
    loop = pts_sorted + [pts_sorted[0]]
    ax.plot([p[0] for p in loop], [p[1] for p in loop], "m-", linewidth=1.5, zorder=3)

    # 2. Internal axes
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k:", linewidth=1, alpha=0.5)
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], "k:", linewidth=1, alpha=0.5)

    # 3. Curved Vesica lens arcs
    d02 = math.hypot(pts_sorted[2][0] - pts_sorted[0][0], pts_sorted[2][1] - pts_sorted[0][1])
    d13 = math.hypot(pts_sorted[3][0] - pts_sorted[1][0], pts_sorted[3][1] - pts_sorted[1][1])

    if d02 >= d13:
        major1, major2 = pts_sorted[0], pts_sorted[2]
        center1, center2 = pts_sorted[1], pts_sorted[3]
    else:
        major1, major2 = pts_sorted[1], pts_sorted[3]
        center1, center2 = pts_sorted[0], pts_sorted[2]

    for c in [center1, center2]:
        r = (math.hypot(major1[0] - c[0], major1[1] - c[1]) + math.hypot(major2[0] - c[0], major2[1] - c[1])) / 2.0
        a1 = math.atan2(major1[1] - c[1], major1[0] - c[0])
        a2 = math.atan2(major2[1] - c[1], major2[0] - c[0])

        diff = (a2 - a1) % (2 * math.pi)
        if diff > math.pi:
            a1, a2 = a2, a1 + 2 * math.pi
        else:
            a2 = a1 + diff

        angles = [a1 + (a2 - a1) * i / 30.0 for i in range(31)]
        arc_x = [c[0] + r * math.cos(a) for a in angles]
        arc_y = [c[1] + r * math.sin(a) for a in angles]
        ax.plot(arc_x, arc_y, "c--", linewidth=1.2, alpha=0.8, zorder=3)

# Calculate score S0 for original points
S0, _, _ = best_vesica_in_n(POINTS_XY)

# ---------------------------------------------------------------
# 3. Monte Carlo Simulation
# ---------------------------------------------------------------
random.seed(42)
N_TRIALS = 1000
random_scores = []

for _ in range(N_TRIALS):
    trial_latlon = [
        (random.uniform(LAT_MIN, LAT_MAX), random.uniform(LON_MIN, LON_MAX))
        for _ in range(5)
    ]
    trial_xy = [latlon_to_xy(lat, lon, ref_lat) for lat, lon in trial_latlon]
    score, _, _ = best_vesica_in_n(trial_xy)
    random_scores.append(score)

pct = 100 * sum(1 for s in random_scores if s >= S0) / N_TRIALS
print(f"Mean % of random layouts matching or exceeding real score (S0={S0:.3f}): {pct:.1f}%")

# ---------------------------------------------------------------
# 4. Plot Likelihood Histogram
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
weights = [100.0 / N_TRIALS] * len(random_scores)
plt.hist(random_scores, bins=30, weights=weights, color="mediumpurple", edgecolor="black", alpha=0.7)
plt.axvline(S0, color="crimson", linestyle="dashed", linewidth=2, label=f"Real Score S0 = {S0:.3f}")
plt.title("Strict Symmetrical Vesica Piscis Monte Carlo Distribution")
plt.xlabel("Vesica Score (Angle x Centering x Side Equality³ x Vesica Ratio)")
plt.ylabel("Chance (%)")
plt.gca().yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
plt.legend(title=f"{pct:.1f}% of random layouts\nmatch or beat S0")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("vesica_strict_score_histogram.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 5. Grid of 9 High-Symmetry Vesica Piscis Layouts
# ---------------------------------------------------------------
SCORE_THRESHOLD = 0.40  # Tightened threshold to guarantee visual symmetry

valid_trials = []
while len(valid_trials) < 9:
    trial_latlon = [
        (random.uniform(LAT_MIN, LAT_MAX), random.uniform(LON_MIN, LON_MAX))
        for _ in range(5)
    ]
    trial_xy = [latlon_to_xy(lat, lon, ref_lat) for lat, lon in trial_latlon]
    score, combo, split = best_vesica_in_n(trial_xy)

    if score >= SCORE_THRESHOLD and split is not None:
        (p1, p2), (p3, p4) = split
        angle = angle_between(p1, p2, p3, p4)
        valid_trials.append({
            "trial_xy": trial_xy,
            "combo": combo,
            "split": split,
            "angle": angle,
            "score": score,
        })

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

for idx, trial in enumerate(valid_trials):
    ax = axes[idx]
    trial_xy = trial["trial_xy"]
    combo = trial["combo"]
    (p1, p2), (p3, p4) = trial["split"]

    used_idx = set(combo)
    used_xy = [trial_xy[i] for i in range(5) if i in used_idx]
    unused_xy = [trial_xy[i] for i in range(5) if i not in used_idx]

    # Plot 4 cross points and 5th unused point
    ax.scatter([p[0] for p in used_xy], [p[1] for p in used_xy], c="crimson", s=40, zorder=4)
    if unused_xy:
        ax.scatter([p[0] for p in unused_xy], [p[1] for p in unused_xy], c="lightgray", s=30, zorder=2)

    # Plot symmetrical Vesica Piscis outline & curved arcs
    plot_vesica_shape(ax, p1, p2, p3, p4)

    ax.set_title(f"Symmetric Vesica #{idx + 1}\n(Score: {trial['score']:.3f}, Angle: {trial['angle']:.1f}°)", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.tight_layout()
plt.savefig("strict_vesica_piscis_grid.png", dpi=150)
plt.show()