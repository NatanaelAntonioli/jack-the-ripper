"""
Jack the Ripper Corpus - Radial Dendrogram
============================================
Reproduces the clustering method from Nini (2018), "An authorship analysis
of the Jack the Ripper letters", Digital Scholarship in the Humanities.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = 'jack_the_ripper_corpus.csv'
OUTPUT_PATH = 'radial_dendrogram.png'
MIN_DOC_FREQ = 2       # a bigram must appear in at least this many texts
N_EXCLUDED_TOP = 8     # exclude the N most frequent bigrams (per the paper)
COLOR_THRESHOLD_FRAC = 0.72  # fraction of max merge height used to color main clusters

# --- DASH & NODE INVALIDATION CONFIG ---
CUSTOM_DASH_NODES = None  # Target specific Node IDs, or None for all poison leaves
INVALIDATION_LEVELS_UP = 2  # 1 = parent (n=2), 2 = grandparent (n=3), etc.

# --- LINE VISIBILITY STYLING ---
SOLID_LINEWIDTH = 2.2     # Bold, prominent solid lines
SOLID_ALPHA = 1.0

DASH_LINEWIDTH = 1.0      # Subdued dashed lines
DASH_ALPHA = 0.45         # De-emphasized opacity
DASH_PATTERN = (3, 3)     # Clean dash pattern

CLUSTER_PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#469990', '#9A6324',
    '#800000', '#808000', '#000075', '#a9a9a9', '#fabed4',
    '#ffd8b1', '#aaffc3', '#dcbeff',
]
ABOVE_THRESHOLD_COLOR = '#4d4d4d'

HIGHLIGHTS = {
    'DEAR BOSS':   'Dear Boss',
    'SAUCY JACKY': 'Saucy Jacky',
    'MIDIAN':      'Moab and Midian',
    'FROM HELL':   'From Hell',
}
EXACT_HIGHLIGHTS = {
    'UNK_240988.txt': 'Text 1 (24 Sep, unsigned)',
    'RI_011088.txt':  'Text 4 (1 Oct)',
}


# ---------------------------------------------------------------------------
# 1-2. Load + tokenize
# ---------------------------------------------------------------------------
def load_corpus(csv_path):
    df = pd.read_csv(csv_path, sep=';', engine='python')
    df = df.dropna(subset=['filename', 'text'])
    df = df[df['filename'].str.strip() != '']
    df = df.reset_index(drop=True)
    df['is_poison'] = df['poison'].notna() if 'poison' in df.columns else False
    return df


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()


def bigrams(tokens):
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


# ---------------------------------------------------------------------------
# 3-4. Build feature set
# ---------------------------------------------------------------------------
def build_features(df):
    df['tokens'] = df['text'].apply(tokenize)
    df['n_tokens'] = df['tokens'].apply(len)
    df['bigram_set'] = df['tokens'].apply(lambda t: set(bigrams(t)))

    doc_freq = Counter()
    for bset in df['bigram_set']:
        doc_freq.update(bset)

    recurring = {bg for bg, c in doc_freq.items() if c >= MIN_DOC_FREQ}
    top_n = [bg for bg, _ in doc_freq.most_common(N_EXCLUDED_TOP)]
    feature_vocab = recurring - set(top_n)

    df['features'] = df['bigram_set'].apply(lambda s: s & feature_vocab)

    print(f"Loaded {len(df)} texts (avg length {df['n_tokens'].mean():.1f} tokens, "
          f"min={df['n_tokens'].min()}, max={df['n_tokens'].max()})")
    print(f"Excluded top {N_EXCLUDED_TOP} bigrams: {top_n}")
    print(f"Feature vocabulary size: {len(feature_vocab)}")
    return df, feature_vocab


# ---------------------------------------------------------------------------
# 5-6. Jaccard distance + Ward clustering
# ---------------------------------------------------------------------------
def cluster_texts(df, feature_vocab):
    vocab_index = {f: i for i, f in enumerate(sorted(feature_vocab))}
    X = np.zeros((len(df), len(vocab_index)), dtype=bool)
    for i, feats in enumerate(df['features']):
        for f in feats:
            X[i, vocab_index[f]] = True

    dist_condensed = pdist(X, metric='jaccard')
    Z = linkage(dist_condensed, method='ward')

    D = squareform(dist_condensed)
    print(f"Median Jaccard distance: {np.median(dist_condensed):.3f}")

    def dist_between(name_substr_a, name_substr_b):
        ia = df.index[df['filename'].str.contains(name_substr_a, case=False)]
        ib = df.index[df['filename'].str.contains(name_substr_b, case=False)]
        if len(ia) and len(ib):
            return D[ia[0], ib[0]]
        return None

    db_sj = dist_between('DEAR BOSS', 'SAUCY JACKY')
    db_mm = dist_between('DEAR BOSS', 'MIDIAN')
    sj_mm = dist_between('SAUCY JACKY', 'MIDIAN')
    print(f"Dear Boss <-> Saucy Jacky:      {db_sj:.3f}  (paper: 0.929)")
    print(f"Dear Boss <-> Moab and Midian:  {db_mm:.3f}  (paper: 0.934)")
    print(f"Saucy Jacky <-> Moab and Midian:{sj_mm:.3f}  (paper: 0.90)")

    return Z


# ---------------------------------------------------------------------------
# 6b. Downward Traversal with Immunity for Highlighted Nodes
# ---------------------------------------------------------------------------
def get_trigger_nodes(df, custom_nodes=None):
    if custom_nodes is not None and len(custom_nodes) > 0:
        return list(custom_nodes)
    return [i for i, is_poison in enumerate(df['is_poison']) if is_poison]


def get_immune_leaves(df):
    immune = set()
    for i, filename in enumerate(df['filename']):
        # RI_011088 (Text 4) is canonical but can still be killed
        if 'RI_011088' in filename:
            continue
        if get_highlight(filename) is not None:
            immune.add(i)
    return immune


def compute_subtree_immunity(Z, n, immune_leaves):
    has_immune = {i: (i in immune_leaves) for i in range(n)}
    for k in range(Z.shape[0]):
        parent = n + k
        a, b = int(Z[k, 0]), int(Z[k, 1])
        has_immune[parent] = has_immune[a] or has_immune[b]
    return has_immune


def compute_dashed_links_downward(Z, trigger_nodes, immune_leaves, levels_up=INVALIDATION_LEVELS_UP):
    n = Z.shape[0] + 1
    has_immune = compute_subtree_immunity(Z, n, immune_leaves)

    parent_of = {}
    children_of = {}
    for k in range(Z.shape[0]):
        parent = n + k
        a, b = int(Z[k, 0]), int(Z[k, 1])
        parent_of[a] = (k, 'a')
        parent_of[b] = (k, 'b')
        children_of[parent] = (a, b, k)

    dashed = set()
    resolved_triggers = []

    def mark_descendants(curr_node):
        if curr_node in children_of:
            a, b, k = children_of[curr_node]
            
            # Branch 'a' becomes dashed ONLY if 'a' contains no immune nodes
            if not has_immune.get(a, False):
                dashed.add((k, 'a'))
            mark_descendants(a)

            # Branch 'b' becomes dashed ONLY if 'b' contains no immune nodes
            if not has_immune.get(b, False):
                dashed.add((k, 'b'))
            mark_descendants(b)

    for node_id in trigger_nodes:
        junction_id = node_id
        
        # Traverse UP by specified number of levels (e.g. 2 for n=3)
        for _ in range(levels_up):
            if junction_id in parent_of:
                k, _ = parent_of[junction_id]
                junction_id = n + k
            else:
                break

        # Only draw an 'X' if the target ancestor node contains NO immune descendants
        if not has_immune.get(junction_id, False):
            resolved_triggers.append(junction_id)

        mark_descendants(junction_id)

    return dashed, resolved_triggers


# ---------------------------------------------------------------------------
# 6c. Color cluster merges & leaf nodes
# ---------------------------------------------------------------------------
def compute_cluster_colors(Z, threshold):
    n = Z.shape[0] + 1
    flat = fcluster(Z, t=threshold, criterion='distance')

    cluster_of = {i: int(flat[i]) for i in range(n)}
    for k in range(Z.shape[0]):
        a, b, h = int(Z[k, 0]), int(Z[k, 1]), Z[k, 2]
        parent_id = n + k
        if h <= threshold:
            cluster_of[parent_id] = cluster_of[a]
        else:
            cluster_of[parent_id] = -1

    unique_clusters = sorted(set(v for v in cluster_of.values() if v != -1))
    palette_map = {cid: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                   for i, cid in enumerate(unique_clusters)}

    link_color = {}
    for k in range(Z.shape[0]):
        cid = cluster_of[n + k]
        link_color[k] = palette_map[cid] if cid != -1 else ABOVE_THRESHOLD_COLOR

    leaf_color = {}
    for i in range(n):
        cid = cluster_of[i]
        leaf_color[i] = palette_map[cid] if cid != -1 else ABOVE_THRESHOLD_COLOR

    return link_color, leaf_color


# ---------------------------------------------------------------------------
# 7. Radial dendrogram plot
# ---------------------------------------------------------------------------
def make_label(filename):
    return filename.split(' - ')[0].replace('.txt', '')


def get_highlight(filename):
    fn_upper = filename.upper()
    for key, val in HIGHLIGHTS.items():
        if key in fn_upper:
            return val
    if filename.strip() in EXACT_HIGHLIGHTS:
        return EXACT_HIGHLIGHTS[filename.strip()]
    return None


def plot_radial_dendrogram(Z, df, output_path):
    codes = df['filename'].apply(make_label).tolist()
    n = len(codes)
    poison_flags = df['is_poison'].tolist()

    max_height = Z[:, 2].max()
    color_threshold = COLOR_THRESHOLD_FRAC * max_height

    order = list(leaves_list(Z))

    x_of = {leaf_id: 5 + 10 * pos for pos, leaf_id in enumerate(order)}
    height_of = {leaf_id: 0.0 for leaf_id in range(n)}
    for k in range(Z.shape[0]):
        a, b, h = int(Z[k, 0]), int(Z[k, 1]), Z[k, 2]
        parent_id = n + k
        x_of[parent_id] = 0.5 * (x_of[a] + x_of[b])
        height_of[parent_id] = h

    immune_leaves = get_immune_leaves(df)
    raw_triggers = get_trigger_nodes(df, CUSTOM_DASH_NODES)
    dashed_links, trigger_junctions = compute_dashed_links_downward(
        Z, raw_triggers, immune_leaves, levels_up=INVALIDATION_LEVELS_UP
    )
    link_color, leaf_color = compute_cluster_colors(Z, color_threshold)

    max_d = Z[:, 2].max()
    max_x = n * 10
    inner_hub = 0.08

    def angle_of(x):
        return (x / max_x) * 2 * np.pi

    def radius_of(d):
        return inner_hub + (1 - inner_hub) * (1 - d / max_d)

    fig = plt.figure(figsize=(18, 18), facecolor='white')
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # --- Draw links ---
    for k in range(Z.shape[0]):
        a, b = int(Z[k, 0]), int(Z[k, 1])
        col = link_color[k]
        xa, xb = x_of[a], x_of[b]
        da, db, dh = height_of[a], height_of[b], Z[k, 2]
        aa, ab = angle_of(xa), angle_of(xb)
        ra, rb, rh = radius_of(da), radius_of(db), radius_of(dh)

        is_dashed_a = (k, 'a') in dashed_links
        is_dashed_b = (k, 'b') in dashed_links

        # Side A (branch going down to child a)
        lw_a = DASH_LINEWIDTH if is_dashed_a else SOLID_LINEWIDTH
        alpha_a = DASH_ALPHA if is_dashed_a else SOLID_ALPHA
        z_a = 2 if is_dashed_a else 3
        line_a, = ax.plot([aa, aa], [ra, rh], color=col, linewidth=lw_a, alpha=alpha_a,
                          linestyle='--' if is_dashed_a else '-',
                          solid_capstyle='round', zorder=z_a)
        if is_dashed_a:
            line_a.set_dashes(DASH_PATTERN)

        # Side B (branch going down to child b)
        lw_b = DASH_LINEWIDTH if is_dashed_b else SOLID_LINEWIDTH
        alpha_b = DASH_ALPHA if is_dashed_b else SOLID_ALPHA
        z_b = 2 if is_dashed_b else 3
        line_b, = ax.plot([ab, ab], [rb, rh], color=col, linewidth=lw_b, alpha=alpha_b,
                          linestyle='--' if is_dashed_b else '-',
                          solid_capstyle='round', zorder=z_b)
        if is_dashed_b:
            line_b.set_dashes(DASH_PATTERN)

        # Connecting arc at merge height rh
        arc_theta = np.linspace(aa, ab, max(6, int(abs(ab - aa) / (2 * np.pi) * 400)))
        ax.plot(arc_theta, [rh] * len(arc_theta), color=col, linewidth=SOLID_LINEWIDTH, alpha=SOLID_ALPHA, zorder=3)

    # --- Draw Red 'X' Markers directly at Invalidation Origin Junctions ---
    for j_id in set(trigger_junctions):
        if j_id in x_of and j_id in height_of:
            j_ang = angle_of(x_of[j_id])
            j_rad = radius_of(height_of[j_id])
            ax.plot(j_ang, j_rad, marker='x', color='#d62728',
                    markersize=12, markeredgewidth=2.6, zorder=6)

    # --- Leaf labels ---
    leaf_x = np.arange(n) * 10 + 5
    leaf_angle = angle_of(leaf_x)
    filename_by_leaf = df['filename'].tolist()
    label_by_leaf = codes

    for pos, leaf_id in enumerate(order):
        ang = leaf_angle[pos]
        lab = label_by_leaf[leaf_id]
        fname = filename_by_leaf[leaf_id]
        is_poison = poison_flags[leaf_id]
        hl_name = get_highlight(fname)
        cluster_c = leaf_color[leaf_id]
        deg = np.degrees(ang)
        rot = deg if not (90 < deg < 270) else deg + 180
        ha = 'left' if not (90 < deg < 270) else 'right'

        tick_style = '--' if is_poison else '-'

        if hl_name:
            ax.plot([ang, ang], [1.0, 1.03], color=cluster_c, linewidth=1.6,
                     linestyle=tick_style, zorder=4)
            ax.text(ang, 1.045, f"{lab}  \u2014  {hl_name}", rotation=rot, rotation_mode='anchor',
                    ha=ha, va='center', fontsize=9, color=cluster_c, weight='bold', zorder=5)
        else:
            tick_color = '#bbbbbb' if is_poison else '#999999'
            ax.plot([ang, ang], [1.0, 1.012], color=tick_color, linewidth=0.5,
                     linestyle=tick_style, zorder=3)
            label_color = '#888888' if is_poison else '#333333'
            ax.text(ang, 1.02, lab, rotation=rot, rotation_mode='anchor',
                    ha=ha, va='center', fontsize=4.6, color=label_color, zorder=3)

    # Central hub circle
    hub_theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(hub_theta, [inner_hub] * len(hub_theta), color='#4d4d4d', linewidth=1.0, zorder=2)

    ax.set_ylim(0, 1.18)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    plt.title("Radial dendrogram of the Jack the Ripper Corpus\n"
              "Ward hierarchical clustering on Jaccard distances (word 2-grams)\n"
              "Red X kills subtree: non-immune descendant branches dashed towards leaves",
              fontsize=14, pad=20)

    legend_elems = [
        Line2D([0], [0], color='#4d4d4d', lw=SOLID_LINEWIDTH, label='Valid branch (solid)'),
        Line2D([0], [0], color='#4d4d4d', lw=DASH_LINEWIDTH, linestyle='--', alpha=DASH_ALPHA, label='Killed subtree (dashed towards leaves)'),
        Line2D([0], [0], marker='x', color='#d62728', lw=0, ms=8, mew=2.5, label='Subtree invalidation origin (X)'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', bbox_to_anchor=(1.02, 1.06),
              fontsize=10, frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight', facecolor='white')
    print(f"Saved dendrogram to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_corpus(CSV_PATH)
    df, feature_vocab = build_features(df)
    Z = cluster_texts(df, feature_vocab)
    plot_radial_dendrogram(Z, df, OUTPUT_PATH)


if __name__ == '__main__':
    main()