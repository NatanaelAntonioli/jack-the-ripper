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
# 2. Geometry & Scoring Functions
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

def arm_pair_score(p1, p2, p3, p4):
    params = line_intersect_params(p1, p2, p3, p4)
    if params is None:
        return 0.0
    t, s = params
    if not (0 <= t <= 1 and 0 <= s <= 1):
        return 0.0
    angle = angle_between(p1, p2, p3, p4)
    angle_score = 1 - abs(angle - 90) / 90
    centering_score = max(0, 1 - (abs(t - 0.5) + abs(s - 0.5)))
    return angle_score * centering_score

def best_cross_in_4(pts):
    a, b, c, d = pts
    splits = [
        ((a, b), (c, d)),
        ((a, c), (b, d)),
        ((a, d), (b, c)),
    ]
    best_score, best_split = -1, None
    for (p1, p2), (p3, p4) in splits:
        sc = arm_pair_score(p1, p2, p3, p4)
        if sc > best_score:
            best_score, best_split = sc, ((p1, p2), (p3, p4))
    return best_score, best_split

def best_cross_in_n(points_xy):
    best_score, best_combo, best_split = -1, None, None
    n = len(points_xy)
    for idx in itertools.combinations(range(n), 4):
        pts = [points_xy[i] for i in idx]
        sc, split = best_cross_in_4(pts)
        if sc > best_score:
            best_score, best_combo, best_split = sc, idx, split
    return best_score, best_combo, best_split

def get_point_to_point_cross(p1, p2, p3, p4):
    arm1 = (p1, p2)
    arm2 = (p3, p4)
    return arm1, arm2

def get_arrow_geometry(p1, p2, p3, p4):
    """
    Identifies the longest arm as the main shaft and the shorter arm as the wings.
    Sets the arrow tip to the shaft endpoint closer to the intersection point,
    ensuring a proper, symmetrical arrowhead shape.
    """
    len1 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    len2 = math.hypot(p4[0] - p3[0], p4[1] - p3[1])

    if len1 >= len2:
        s1, s2 = p1, p2
        w1, w2 = p3, p4
    else:
        s1, s2 = p3, p4
        w1, w2 = p1, p2

    params = line_intersect_params(s1, s2, w1, w2)
    if params is not None:
        t, _ = params
        if t >= 0.5:
            tip, tail = s2, s1
        else:
            tip, tail = s1, s2
    else:
        tip, tail = s2, s1

    return tail, tip, w1, w2

# Calculate score S0 for original points
S0, _, _ = best_cross_in_n(POINTS_XY)

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
    score, _, _ = best_cross_in_n(trial_xy)
    random_scores.append(score)

pct = 100 * sum(1 for s in random_scores if s >= S0) / N_TRIALS
print(f"Mean % of random crosses matching or exceeding real score: {pct:.1f}%")

# ---------------------------------------------------------------
# 4. Plot Likelihood Histogram
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
weights = [100.0 / N_TRIALS] * len(random_scores)
plt.hist(random_scores, bins=30, weights=weights, color="skyblue", edgecolor="black", alpha=0.7)
plt.axvline(S0, color="crimson", linestyle="dashed", linewidth=2, label=f"Real Score S0 = {S0:.3f}")
plt.title("Monte Carlo Score Distribution (Chance of Occurring)")
plt.xlabel("Cross Score")
plt.ylabel("Chance (%)")
plt.gca().yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
plt.legend(title=f"{pct:.1f}% of random layouts\nmatch or beat S0")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("score_histogram.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 5. Store Valid Random Layouts
# ---------------------------------------------------------------
ANGLE_TOLERANCE = 15
SCORE_THRESHOLD = 0.5

valid_trials = []

while len(valid_trials) < 9:
    trial_latlon = [
        (random.uniform(LAT_MIN, LAT_MAX), random.uniform(LON_MIN, LON_MAX))
        for _ in range(5)
    ]
    trial_xy = [latlon_to_xy(lat, lon, ref_lat) for lat, lon in trial_latlon]
    score, combo, split = best_cross_in_n(trial_xy)

    if score > 0 and split is not None:
        (p1, p2), (p3, p4) = split
        angle = angle_between(p1, p2, p3, p4)

        if score >= SCORE_THRESHOLD and abs(angle - 90) <= ANGLE_TOLERANCE:
            valid_trials.append({
                "trial_xy": trial_xy,
                "combo": combo,
                "split": split,
                "angle": angle,
                "score": score,
            })

# ---------------------------------------------------------------
# 5a. Grid 1: Cross Topology (3x3)
# ---------------------------------------------------------------
fig1, axes1 = plt.subplots(3, 3, figsize=(10, 10))
axes1 = axes1.flatten()

for idx, trial in enumerate(valid_trials):
    ax = axes1[idx]
    trial_xy = trial["trial_xy"]
    combo = trial["combo"]
    (p1, p2), (p3, p4) = trial["split"]

    used_idx = set(combo)
    used_xy = [trial_xy[i] for i in range(5) if i in used_idx]
    unused_xy = [trial_xy[i] for i in range(5) if i not in used_idx]

    ax.scatter([p[0] for p in used_xy], [p[1] for p in used_xy], c="crimson", s=40, zorder=3)
    if unused_xy:
        ax.scatter([p[0] for p in unused_xy], [p[1] for p in unused_xy], c="lightgray", s=30, zorder=2)

    arm1, arm2 = get_point_to_point_cross(p1, p2, p3, p4)
    ax.plot([arm1[0][0], arm1[1][0]], [arm1[0][1], arm1[1][1]], "b-", linewidth=1.5)
    ax.plot([arm2[0][0], arm2[1][0]], [arm2[0][1], arm2[1][1]], "g-", linewidth=1.5)

    ax.set_title(f"Cross #{idx + 1}\n(Angle: {trial['angle']:.1f}°, Score: {trial['score']:.3f})", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.tight_layout()
plt.savefig("near_90deg_crosses_grid.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 5b. Grid 2: Fixed Arrow Topology (3x3)
# ---------------------------------------------------------------
fig2, axes2 = plt.subplots(3, 3, figsize=(10, 10))
axes2 = axes2.flatten()

for idx, trial in enumerate(valid_trials):
    ax = axes2[idx]
    trial_xy = trial["trial_xy"]
    combo = trial["combo"]
    (p1, p2), (p3, p4) = trial["split"]

    used_idx = set(combo)
    used_xy = [trial_xy[i] for i in range(5) if i in used_idx]
    unused_xy = [trial_xy[i] for i in range(5) if i not in used_idx]

    ax.scatter([p[0] for p in used_xy], [p[1] for p in used_xy], c="crimson", s=40, zorder=3)
    if unused_xy:
        ax.scatter([p[0] for p in unused_xy], [p[1] for p in unused_xy], c="lightgray", s=30, zorder=2)

    # Calculate robust arrow geometry
    tail, tip, w1, w2 = get_arrow_geometry(p1, p2, p3, p4)

    # Main Arrow Shaft (Solid Blue Line)
    ax.plot([tail[0], tip[0]], [tail[1], tip[1]], "b-", linewidth=1.5)
    # Arrow Wings (Dashed Magenta Lines connecting crossbar ends to tip)
    ax.plot([w1[0], tip[0]], [w1[1], tip[1]], "m--", linewidth=1.5)
    ax.plot([w2[0], tip[0]], [w2[1], tip[1]], "m--", linewidth=1.5)

    ax.set_title(f"Arrow #{idx + 1}\n(Angle: {trial['angle']:.1f}°, Score: {trial['score']:.3f})", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.tight_layout()
plt.savefig("near_90deg_arrows_grid.png", dpi=150)
plt.show()