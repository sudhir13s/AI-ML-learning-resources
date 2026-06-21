"""Association Rule Learning concept-page diagrams (muted palette, parallel scale). REAL measured.

Four figures for 04. Unsupervised_Learning/concepts/11-Association-Rule-Learning.md:
  1. apriori_lattice.png    -- the itemset lattice for items {A,B,C,D} with the
     anti-monotone (downward-closure) pruning made visible: {A,B} is infrequent, so
     EVERY superset ({A,B,C}, {A,B,D}, {A,B,C,D}) is pruned without ever being
     counted. Frequent / infrequent / pruned nodes colour-coded; the killed edges
     dashed. Supports computed on a real 8-transaction DB.
  2. support_conf_lift.png  -- a measured scatter of every rule mined from a small
     grocery basket DB: x = confidence, y = lift, size = support. The lift=1
     independence baseline drawn, and the "confidence trap" zone (high confidence,
     lift<=1 -- worthless) shaded. Real mlxtend output.
  3. fp_tree.png            -- an FP-tree schematic built from a small DB (5 txns)
     after frequency-descending item reordering: the shared prefix path is the whole
     point (compression), with the header-table node-links drawn dashed.
  4. top_rules_lift.png     -- a measured horizontal bar chart of the top association
     rules by lift on the grocery DB, each bar annotated with support & confidence.

Run with: /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_association_rules_diagrams.py
"""
import os
from itertools import combinations, chain

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

OUT = os.path.join(os.path.dirname(__file__), "..", "04. Unsupervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------------------
# A small grocery basket database reused by figures 2 and 4.
# ----------------------------------------------------------------------------
GROCERIES = [
    ["bread", "milk"],
    ["bread", "diapers", "beer", "eggs"],
    ["milk", "diapers", "beer", "cola"],
    ["bread", "milk", "diapers", "beer"],
    ["bread", "milk", "diapers", "cola"],
    ["bread", "milk", "diapers", "beer"],
    ["milk", "diapers", "cola"],
    ["bread", "beer"],
    ["bread", "milk", "beer", "diapers"],
    ["milk", "cola"],
]


def _basket_rules(min_support=0.2, min_conf=0.3):
    te = TransactionEncoder()
    arr = te.fit(GROCERIES).transform(GROCERIES)
    df = pd.DataFrame(arr, columns=te.columns_)
    freq = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    return rules


# ----------------------------------------------------------------------------
# Figure 1 -- Apriori lattice with anti-monotone pruning.
# ----------------------------------------------------------------------------
def apriori_lattice():
    # 8-transaction DB over items A,B,C,D; supports counted exactly.
    # Designed so {A,B} is INFREQUENT (A and B co-occur only once -> support 1/8),
    # which makes the downward-closure pruning of all {A,B,*} supersets real.
    txns = [
        {"A", "C", "D"},
        {"B", "C"},
        {"A", "C", "D"},
        {"B", "C", "D"},
        {"A", "C"},
        {"A", "C", "D"},
        {"B", "C", "D"},
        {"A", "B", "C"},   # the single transaction where A and B co-occur
    ]
    n = len(txns)
    items = ["A", "B", "C", "D"]
    min_sup = 0.25  # = 2 / 8

    def support(s):
        s = set(s)
        return sum(1 for t in txns if s.issubset(t)) / n

    # Build the lattice level by level WITH apriori pruning so we know which nodes
    # were pruned (a candidate is pruned if ANY immediate subset is infrequent).
    levels = {1: [(it,) for it in items]}
    frequent = set()
    status = {}  # itemset(frozenset) -> "frequent" | "infrequent" | "pruned"

    def evaluate(level_sets, prev_frequent):
        out = []
        for s in level_sets:
            fs = frozenset(s)
            # pruning: every (k-1)-subset must be frequent
            subs = [frozenset(c) for c in combinations(s, len(s) - 1)] if len(s) > 1 else []
            if subs and any(sub not in prev_frequent for sub in subs):
                status[fs] = "pruned"
                continue
            if support(s) >= min_sup:
                status[fs] = "frequent"
                frequent.add(fs)
                out.append(fs)
            else:
                status[fs] = "infrequent"
        return out

    f1 = evaluate(levels[1], set())
    cand2 = [frozenset(c) for c in combinations(sorted({i for fs in f1 for i in fs}), 2)]
    f2 = evaluate(cand2, set(f1))
    # candidate 3-itemsets generated by joining frequent 2-itemsets
    cand3 = set()
    f2l = list(f2)
    for i in range(len(f2l)):
        for j in range(i + 1, len(f2l)):
            u = f2l[i] | f2l[j]
            if len(u) == 3:
                cand3.add(u)
    f3 = evaluate(sorted(cand3, key=lambda x: tuple(sorted(x))), set(f2))
    # also surface the supersets that pruning let us skip -> mark them "pruned"
    for s in combinations(items, 3):
        fs = frozenset(s)
        if fs not in status:
            status[fs] = "pruned"
    quad = frozenset(items)
    status.setdefault(quad, "pruned")

    # layout: 4 rows (level 1..4), centered
    def lab(fs):
        return "{" + ",".join(sorted(fs)) + "}"

    rows = {
        1: [frozenset((it,)) for it in items],
        2: [frozenset(c) for c in combinations(items, 2)],
        3: [frozenset(c) for c in combinations(items, 3)],
        4: [quad],
    }
    pos = {}
    for lvl, nodes in rows.items():
        y = 4 - lvl
        xs = np.linspace(-len(nodes) / 2.0, len(nodes) / 2.0, len(nodes))
        for x, fs in zip(xs, nodes):
            pos[fs] = (x, y)

    color = {
        "frequent": GREEN,
        "infrequent": RED,
        "pruned": SLATE,
    }

    fig, ax = plt.subplots(figsize=(11, 7.4))
    # edges: subset -> superset (immediate)
    for lvl in (1, 2, 3):
        for a in rows[lvl]:
            for b in rows[lvl + 1]:
                if a < b and len(b - a) == 1:
                    killed = status.get(b) == "pruned" or status.get(a) in ("infrequent",)
                    ax.plot(
                        [pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                        color="#888" if not killed else RED,
                        lw=1.0 if not killed else 1.4,
                        ls="-" if not killed else (0, (4, 3)),
                        alpha=0.45 if not killed else 0.8, zorder=1,
                    )
    for fs, (x, y) in pos.items():
        st = status.get(fs, "pruned")
        sup = support(fs)
        txt = lab(fs)
        sub = f"s={sup:.2f}" if st != "pruned" else "pruned"
        ax.scatter([x], [y], s=2300, c=color[st], edgecolors="white", linewidths=1.6, zorder=2)
        ax.text(x, y + 0.07, txt, ha="center", va="center", color="white",
                fontsize=11, fontweight="bold", zorder=3)
        ax.text(x, y - 0.16, sub, ha="center", va="center", color="white",
                fontsize=8.5, zorder=3)

    # legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markersize=13,
               label="frequent (support ≥ 0.25)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=13,
               label="infrequent (counted, support < 0.25)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SLATE, markersize=13,
               label="PRUNED — never counted (a subset was infrequent)"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=1, frameon=False, fontsize=10)
    ax.annotate("{A,B} is infrequent →\nanti-monotonicity prunes\nall its supersets",
                xy=pos[frozenset({"A", "B"})], xytext=(2.4, 2.05),
                fontsize=9.5, color=RED, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    for y_lvl, name in [(3, "L1 (items)"), (2, "L2 (pairs)"), (1, "L3 (triples)"), (0, "L4 (all)")]:
        ax.text(-3.55, y_lvl, name, ha="left", va="center", fontsize=9.5,
                color=NAVY, fontweight="bold")
    ax.set_title("Apriori itemset lattice — downward-closure pruning in action\n"
                 "(8-transaction DB, min_support = 0.25)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(-0.7, 3.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "apriori_lattice.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote apriori_lattice.png  | frequent itemsets:",
          sorted(("".join(sorted(fs)) for fs in frequent), key=lambda s: (len(s), s)))


# ----------------------------------------------------------------------------
# Figure 2 -- support / confidence / lift scatter with the confidence trap.
# ----------------------------------------------------------------------------
def support_conf_lift():
    rules = _basket_rules(min_support=0.2, min_conf=0.2)
    conf = rules["confidence"].values
    lift = rules["lift"].values
    sup = rules["support"].values

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.set_ylim(lift.min() - 0.07, lift.max() + 0.13)
    # confidence-trap zone: high confidence but lift <= 1
    ax.axhspan(lift.min() - 0.07, 1.0, xmin=0.0, xmax=1.0, color=RED, alpha=0.07, zorder=0)
    ax.axhline(1.0, color=RED, ls="--", lw=1.6, zorder=1)
    ax.text(0.985, 1.005, "lift = 1  (X and Y independent — the baseline)",
            ha="right", va="bottom", color=RED, fontsize=10, fontweight="bold")

    sizes = 60 + sup * 1600
    good = lift > 1.0
    ax.scatter(conf[good], lift[good], s=sizes[good], c=GREEN, alpha=0.78,
               edgecolors="white", linewidths=1.0, label="lift > 1  (positive association)")
    ax.scatter(conf[~good], lift[~good], s=sizes[~good], c=RED, alpha=0.78,
               edgecolors="white", linewidths=1.0, label="lift ≤ 1  (the confidence trap)")

    # annotate the worst trap: highest-confidence rule with lift STRICTLY < 1
    strict_trap = np.where(lift < 0.999)[0]
    if len(strict_trap):
        worst = strict_trap[np.argmax(conf[strict_trap])]
        a = ", ".join(list(rules.iloc[worst]["antecedents"]))
        c = ", ".join(list(rules.iloc[worst]["consequents"]))
        ax.annotate(f"{{{a}}} → {{{c}}}\nconf={conf[worst]:.2f} but lift={lift[worst]:.2f} < 1\n"
                    f"→ high confidence, NEGATIVE signal",
                    xy=(conf[worst], lift[worst]), xytext=(conf[worst] + 0.02, lift[worst] - 0.14),
                    fontsize=9, color=RED, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    # annotate the best rule by lift (placed low so it never collides with the title)
    best = int(np.argmax(lift))
    a = ", ".join(list(rules.iloc[best]["antecedents"]))
    c = ", ".join(list(rules.iloc[best]["consequents"]))
    ax.annotate(f"{{{a}}} → {{{c}}}\nlift={lift[best]:.2f} (strongest)",
                xy=(conf[best], lift[best]), xytext=(conf[best] + 0.05, lift[best] - 0.06),
                fontsize=9, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))

    _despine(ax)
    ax.set_xlabel("confidence  =  P(Y | X)", fontsize=11)
    ax.set_ylabel("lift  =  confidence / support(Y)", fontsize=11)
    ax.set_title("Every mined rule on the confidence–lift plane\n"
                 "(bubble size = support; lift is the metric that filters coincidences)",
                 fontsize=13.5, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "support_conf_lift.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote support_conf_lift.png | {len(rules)} rules; "
          f"{(~good).sum()} in the trap (lift<=1)")


# ----------------------------------------------------------------------------
# Figure 3 -- FP-tree schematic (built from a small DB).
# ----------------------------------------------------------------------------
def fp_tree():
    # Classic small DB (Han et al. style). Items sorted by descending frequency.
    db = [
        ["f", "a", "c", "d", "g", "i", "m", "p"],
        ["a", "b", "c", "f", "l", "m", "o"],
        ["b", "f", "h", "j", "o"],
        ["b", "c", "k", "s", "p"],
        ["a", "f", "c", "e", "l", "p", "m", "n"],
    ]
    min_sup_count = 3
    # frequency of single items
    from collections import Counter
    cnt = Counter(i for t in db for i in t)
    freq_items = [i for i, c in cnt.items() if c >= min_sup_count]
    order = sorted(freq_items, key=lambda i: (-cnt[i], i))  # descending freq
    # reorder each transaction, dropping infrequent items
    ordered = [[i for i in order if i in set(t)] for t in db]

    # Build FP-tree
    class Node:
        def __init__(self, item, parent):
            self.item = item
            self.count = 0
            self.parent = parent
            self.children = {}

    root = Node(None, None)
    nodes_by_item = {i: [] for i in order}
    for t in ordered:
        cur = root
        for it in t:
            if it not in cur.children:
                cur.children[it] = Node(it, cur)
                nodes_by_item[it].append(cur.children[it])
            cur = cur.children[it]
            cur.count += 1

    # layout the tree
    pos = {}
    xcounter = [0.0]

    def layout(node, depth):
        if not node.children:
            x = xcounter[0]
            xcounter[0] += 1.0
            pos[id(node)] = (x, -depth)
            return x
        xs = [layout(c, depth + 1) for c in node.children.values()]
        x = sum(xs) / len(xs)
        pos[id(node)] = (x, -depth)
        return x

    layout(root, 0)

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    # edges
    def draw_edges(node):
        for c in node.children.values():
            x0, y0 = pos[id(node)]
            x1, y1 = pos[id(c)]
            ax.plot([x0, x1], [y0, y1], color="#888", lw=1.3, zorder=1)
            draw_edges(c)
    draw_edges(root)

    # nodes
    def draw_nodes(node):
        x, y = pos[id(node)]
        if node.item is None:
            ax.scatter([x], [y], s=1500, c=SLATE, edgecolors="white", linewidths=1.5, zorder=2)
            ax.text(x, y, "root", ha="center", va="center", color="white",
                    fontsize=9.5, fontweight="bold", zorder=3)
        else:
            ax.scatter([x], [y], s=1600, c=BLUE, edgecolors="white", linewidths=1.5, zorder=2)
            ax.text(x, y, f"{node.item}:{node.count}", ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold", zorder=3)
        for c in node.children.values():
            draw_nodes(c)
    draw_nodes(root)

    # header table (node-links) -- draw dashed links between same-item nodes
    for it in order:
        chain_nodes = nodes_by_item[it]
        for a, b in zip(chain_nodes, chain_nodes[1:]):
            xa, ya = pos[id(a)]
            xb, yb = pos[id(b)]
            ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.3,
                                        ls=(0, (3, 2)), connectionstyle="arc3,rad=-0.25"),
                        zorder=1)
    # header-table list on the left
    htxt = "header table\n(by ↓freq)\n" + "\n".join(f"{i} : {cnt[i]}" for i in order)
    ax.text(-1.4, -1.5, htxt, ha="left", va="top", fontsize=9.5, color=NAVY,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=NAVY, lw=1.2))

    ax.set_title("FP-tree: the database compressed into a shared-prefix tree\n"
                 "(5 transactions, min_support_count = 3; amber dashed = header node-links)",
                 fontsize=13.5, fontweight="bold")
    ax.set_xlim(-1.6, xcounter[0] + 0.3)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fp_tree.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote fp_tree.png | item order (↓freq):", order)


# ----------------------------------------------------------------------------
# Figure 4 -- top rules by lift (measured bar chart).
# ----------------------------------------------------------------------------
def top_rules_lift():
    rules = _basket_rules(min_support=0.2, min_conf=0.3)
    rules = rules.sort_values("lift", ascending=False).head(10).iloc[::-1]
    labels = [
        "{" + ", ".join(sorted(a)) + "} → {" + ", ".join(sorted(c)) + "}"
        for a, c in zip(rules["antecedents"], rules["consequents"])
    ]
    lift = rules["lift"].values
    sup = rules["support"].values
    conf = rules["confidence"].values

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    y = np.arange(len(labels))
    colors = [GREEN if l > 1 else RED for l in lift]
    ax.barh(y, lift, color=colors, edgecolor="white", height=0.66)
    ax.axvline(1.0, color=RED, ls="--", lw=1.5)
    ax.text(1.0, -0.85, "lift = 1 (independence)", color=RED,
            fontsize=9.5, fontweight="bold", va="center", ha="center")
    for yi, (l, s, c) in enumerate(zip(lift, sup, conf)):
        ax.text(l + 0.02, yi, f"  supp={s:.2f}, conf={c:.2f}", va="center",
                fontsize=8.8, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    _despine(ax)
    ax.set_xlabel("lift", fontsize=11)
    ax.set_title("Top association rules by lift — grocery basket DB (mlxtend, measured)\n"
                 "ranked by lift; support & confidence annotated",
                 fontsize=13.5, fontweight="bold")
    ax.set_xlim(0, lift.max() * 1.35)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "top_rules_lift.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote top_rules_lift.png | top lift = {lift.max():.2f}")


if __name__ == "__main__":
    apriori_lattice()
    support_conf_lift()
    fp_tree()
    top_rules_lift()
    print("ALL DIAGRAMS DONE ->", OUT)
