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
# 2. Geometry & Pentagram Scoring Engine
# ---------------------------------------------------------------
def pentagram_score(pts):
    """
    Evaluates how closely 5 points form a regular pentagram / 5-pointed star:
    1. Centroid C: Center of mass of all 5 points.
    2. Radial Consistency: Penalizes variation in distance from C to each point.
    3. Angular Uniformity: Penalizes deviation from ideal 72° spacing around C.
    4. Star Edge Equality: Squared ratio of shortest to longest star edge.
    """
    if len(pts) != 5:
        return 0.0

    # Calculate centroid
    cx = sum(p[0] for p in pts) / 5.0
    cy = sum(p[1] for p in pts) / 5.0

    # Sort points counterclockwise by angle around centroid
    sorted_pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    # 1. Radial distance score (min_r / max_r)
    radii = [math.hypot(p[0] - cx, p[1] - cy) for p in sorted_pts]
    min_r, max_r = min(radii), max(radii)
    if max_r < 1e-6:
        return 0.0
    radial_score = min_r / max_r

    # 2. Angular spacing score (ideal angle diff = 72 degrees)
    angles = [math.degrees(math.atan2(p[1] - cy, p[0] - cx)) % 360 for p in sorted_pts]
    angle_diffs = [(angles[(i + 1) % 5] - angles[i]) % 360 for i in range(5)]
    avg_angle_dev = sum(abs(d - 72.0) for d in angle_diffs) / 5.0
    angular_score = max(0.0, 1.0 - (avg_angle_dev / 72.0))

    # 3. Star edge length equality score (connecting alternating vertices: 0->2->4->1->3->0)
    star_indices = [0, 2, 4, 1, 3]
    edges = [
        math.hypot(
            sorted_pts[star_indices[i]][0] - sorted_pts[star_indices[(i + 1) % 5]][0],
            sorted_pts[star_indices[i]][1] - sorted_pts[star_indices[(i + 1) % 5]][1],
        )
        for i in range(5)
    ]
    min_edge, max_edge = min(edges), max(edges)
    if max_edge < 1e-6:
        return 0.0
    edge_score = (min_edge / max_edge) ** 2

    return radial_score * angular_score * edge_score

def plot_pentagram_shape(ax, pts):
    """
    Renders the star connections, outer perimeter polygon, and centroid.
    """
    cx = sum(p[0] for p in pts) / 5.0
    cy = sum(p[1] for p in pts) / 5.0

    sorted_pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    # Outer convex pentagon frame (faint)
    outer_loop = sorted_pts + [sorted_pts[0]]
    ax.plot([p[0] for p in outer_loop], [p[1] for p in outer_loop], "gray", linestyle=":", linewidth=1, alpha=0.6)

    # 5-Pointed Star (Pentagram chords)
    star_indices = [0, 2, 4, 1, 3, 0]
    star_x = [sorted_pts[i][0] for i in star_indices]
    star_y = [sorted_pts[i][1] for i in star_indices]
    ax.plot(star_x, star_y, "crimson", linewidth=1.5, zorder=3)

    # Plot centroid
    ax.scatter([cx], [cy], c="black", marker="x", s=20, zorder=4)

# Calculate real score S0 for original 5 points
S0 = pentagram_score(POINTS_XY)

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
    score = pentagram_score(trial_xy)
    random_scores.append(score)

pct = 100 * sum(1 for s in random_scores if s >= S0) / N_TRIALS
print(f"Real Score (S0): {S0:.3f}")
print(f"Mean % of random layouts matching or exceeding real score: {pct:.1f}%")

# ---------------------------------------------------------------
# 4. Plot Likelihood Histogram
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
weights = [100.0 / N_TRIALS] * len(random_scores)
plt.hist(random_scores, bins=30, weights=weights, color="teal", edgecolor="black", alpha=0.7)
plt.axvline(S0, color="crimson", linestyle="dashed", linewidth=2, label=f"Real Score S0 = {S0:.3f}")
plt.title("Pentagram Symmetry Monte Carlo Distribution")
plt.xlabel("Pentagram Score (Radial x Angular x Edge Equality²)")
plt.ylabel("Chance (%)")
plt.gca().yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
plt.legend(title=f"{pct:.1f}% of random layouts\nmatch or beat S0")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("pentagram_score_histogram.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 5. Grid of 9 High-Symmetry Pentagram Layouts
# ---------------------------------------------------------------
SCORE_THRESHOLD = 0.40  # Threshold to guarantee visual pentagram symmetry

valid_trials = []
while len(valid_trials) < 9:
    trial_latlon = [
        (random.uniform(LAT_MIN, LAT_MAX), random.uniform(LON_MIN, LON_MAX))
        for _ in range(5)
    ]
    trial_xy = [latlon_to_xy(lat, lon, ref_lat) for lat, lon in trial_latlon]
    score = pentagram_score(trial_xy)

    if score >= SCORE_THRESHOLD:
        valid_trials.append({"trial_xy": trial_xy, "score": score})

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

for idx, trial in enumerate(valid_trials):
    ax = axes[idx]
    pts = trial["trial_xy"]

    # Plot 5 star vertices
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], c="crimson", s=40, zorder=5)

    # Render Pentagram star lines
    plot_pentagram_shape(ax, pts)

    ax.set_title(f"Symmetric Pentagram #{idx + 1}\n(Score: {trial['score']:.3f})", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.tight_layout()
plt.savefig("pentagram_grid.png", dpi=150)
plt.show()