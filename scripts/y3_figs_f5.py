#!/usr/bin/env python
"""Paper Y3 -- Figure 1: the SURGE correction loop (full text width, 16.46 cm).

One rectangular flowline and nothing else.  The operations row runs left to
right; a single arrow drops at the far right into the learning row; the
learning row runs right to left underneath it; a single arrow rises at the far
left back into the scorer.  Six stage boxes of one width and one height on an
explicit three-column grid, six arrows of one style, two band headers and two
annotations.

Geometry is *verified*, not assumed.  After drawing, the script checks that

  * no two drawn arrow segments intersect,
  * no arrow segment intersects any stage box,
  * no text block overlaps a box it does not belong to, any arrow, or any
    other text block,
  * every box's own text fits inside it,

and prints the result.  Run with an optional argument to also write a 300 dpi
PNG for visual inspection:

    python scripts/y3_figs_f5.py [inspect.png]
"""
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path("paper/figures/f5_schematic.pdf")
CM = 1 / 2.54
PT = 2.54 / 72.0                       # one typographic point, in centimetres

# ---------------------------------------------------------------------------
# Fonts: TeX Gyre Termes (metric-identical Times) for text, STIX for mathtext,
# so the figure's type matches the manuscript's newtxtext/newtxmath.
# ---------------------------------------------------------------------------
_TG = Path.home() / ".TinyTeX/texmf-dist/fonts/opentype/public/tex-gyre"
for _f in ("texgyretermes-regular.otf", "texgyretermes-bold.otf",
           "texgyretermes-italic.otf", "texgyretermes-bolditalic.otf"):
    if (_TG / _f).exists():
        fm.fontManager.addfont(str(_TG / _f))
_SERIF = "TeX Gyre Termes" if (_TG / "texgyretermes-regular.otf").exists() else "Nimbus Roman"
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": [_SERIF, "Nimbus Roman", "Liberation Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
})

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
INK = "#1A1A1A"                        # all text and all lines
FILL_OPS = "#DCEAF6"                   # accent 1: runs at every dispatch event
FILL_LRN = "#F7DFC9"                   # accent 2: runs on the logged reviews

FS_HEAD, FS_TITLE, FS_BODY = 10.0, 10.0, 9.0
LW_BOX, LW_ARROW = 0.9, 1.1            # hairline frame / standard connector
ARROW_MS = 11
SHRINK_PT = 2.0                        # arrow standoff from a box edge, points
SHRINK = SHRINK_PT * PT

# ---------------------------------------------------------------------------
# Grid.  Three columns, two rows, one inter-row band.
# ---------------------------------------------------------------------------
W = 16.46
MARGIN = 0.14
BW, GAPX = 4.80, 0.89                  # 3*4.80 + 2*0.89 = 16.18 = W - 2*MARGIN
BH = 2.78
BAND = 1.95                            # clear space between the two rows
HEAD_H, HEAD_GAP = 0.38, 0.16
TITLE_DY, BODY_DY = 0.47, 1.72         # drop from a box's top edge

LRN_Y0 = MARGIN + HEAD_H + HEAD_GAP                # 0.68
LRN_Y1 = LRN_Y0 + BH                               # 3.46
OPS_Y0 = LRN_Y1 + BAND                             # 5.41
OPS_Y1 = OPS_Y0 + BH                               # 8.19
H = OPS_Y1 + HEAD_GAP + HEAD_H + MARGIN            # 8.87

COL = [MARGIN + i * (BW + GAPX) for i in range(3)]  # 0.14, 5.83, 11.52
CX = [x + BW / 2 for x in COL]                      # 2.54, 8.23, 13.92
OPS_MID = OPS_Y0 + BH / 2
LRN_MID = LRN_Y0 + BH / 2
BAND_MID = (LRN_Y1 + OPS_Y0) / 2

fig = plt.figure(figsize=(W * CM, H * CM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

BOXES = []      # (name, x, y, w, h)
SEGS = []       # (name, (x0, y0), (x1, y1))  -- drawn extent, standoff applied
TEXTS = []      # (name, owner_box_or_None, Text)


def box(name, x, y, fill):
    ax.add_patch(Rectangle((x, y), BW, BH, fc=fill, ec=INK, lw=LW_BOX, zorder=2))
    BOXES.append((name, x, y, BW, BH))
    return x, y


def txt(name, x, y, s, fs=FS_BODY, weight="normal", ha="center", va="center",
        owner=None, style="normal"):
    t = ax.text(x, y, s, fontsize=fs, fontweight=weight, color=INK, ha=ha, va=va,
                style=style, zorder=4, linespacing=1.30)
    TEXTS.append((name, owner, t))
    return t


def stage(name, col, y, fill, title, body):
    """One stage of the flow: same width, same height, centred Title Case
    title over a centred four-line body."""
    x, _ = box(name, COL[col], y, fill)
    txt(f"{name}.title", x + BW / 2, y + BH - TITLE_DY, title, FS_TITLE, "bold",
        owner=name)
    txt(f"{name}.body", x + BW / 2, y + BH - BODY_DY, body, FS_BODY, owner=name)


def arrow(name, p0, p1):
    """One arrow style throughout: ink, standard weight, one arrowhead."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ARROW_MS,
                                 lw=LW_ARROW, color=INK, zorder=3,
                                 shrinkA=SHRINK_PT, shrinkB=SHRINK_PT))
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = (dx ** 2 + dy ** 2) ** 0.5
    ux, uy = dx / n, dy / n
    SEGS.append((name, (p0[0] + ux * SHRINK, p0[1] + uy * SHRINK),
                 (p1[0] - ux * SHRINK, p1[1] - uy * SHRINK)))


# ============================ operations row ===============================
txt("head.ops", W / 2, OPS_Y1 + HEAD_GAP + HEAD_H / 2,
    "Operations: Dispatch Under a Review Budget", FS_HEAD, "bold")

stage("ops1", 0, OPS_Y0, FILL_OPS, "Score and Recommend",
      "Open work orders are ranked\n"
      "by the tuned Apparent Tardiness\n"
      "Cost rule, scored with the\n"
      "corrected class $\\hat{c}$, not $c$.")

stage("ops2", 1, OPS_Y0, FILL_OPS, "Supervisor Review",
      "A share $\\rho$ is reviewed: those\n"
      "SURGE cannot settle. The\n"
      "supervisor knows the true urgency\n"
      "and confirms or overrides.")

stage("ops3", 2, OPS_Y0, FILL_OPS, "Execute and Validate",
      "The crew starts the chosen order.\n"
      "An independent validator scores\n"
      "the schedule on true weighted\n"
      "tardiness $\\mathrm{TWT}^{*}(w^{*}\\!,d^{*})$.")

arrow("ops1->ops2", (COL[0] + BW, OPS_MID), (COL[1], OPS_MID))
arrow("ops2->ops3", (COL[1] + BW, OPS_MID), (COL[2], OPS_MID))

# ============================ the two turns ================================
LEAD = 0.46                            # annotation line pitch, centimetres

arrow("drop", (CX[2], OPS_Y0), (CX[2], LRN_Y1))
txt("lab.drop", CX[2] - 0.20, BAND_MID,
    "the run’s reviewed decisions", FS_BODY, ha="right", va="center")

arrow("rise", (CX[0], LRN_Y1), (CX[0], OPS_Y0))
txt("lab.surge1", CX[0] + 0.20, BAND_MID + LEAD,
    "SURGE correction", FS_BODY, "bold", ha="left", va="center")
txt("lab.surge2", CX[0] + 0.20, BAND_MID,
    "$\\hat{c}=\\mathrm{clip}(c-\\hat{s}(x),\\,1,\\,4)$", FS_BODY, ha="left", va="center")
txt("lab.surge3", CX[0] + 0.20, BAND_MID - LEAD,
    "applied to every decision, reviewed or not", FS_BODY, "bold",
    ha="left", va="center")

# ============================ learning row =================================
stage("lrn3", 2, LRN_Y0, FILL_LRN, "Override Log",
      "Every reviewed decision is\n"
      "recorded: the rule’s pick, the\n"
      "supervisor’s pick, and whether\n"
      "it was confirmed or overridden.")

stage("lrn2", 1, LRN_Y0, FILL_LRN, "Weak Pairwise Labels",
      "An override says one order was\n"
      "more urgent than another; a\n"
      "confirmation says the pick\n"
      "needed no correction.")

stage("lrn1", 0, LRN_Y0, FILL_LRN, "Urgency Estimator",
      "$\\hat{s}(x)$ is refitted from these\n"
      "labels in seconds, using\n"
      "observable features only, with\n"
      "no reinforcement learning.")

arrow("lrn3->lrn2", (COL[2], LRN_MID), (COL[1] + BW, LRN_MID))
arrow("lrn2->lrn1", (COL[1], LRN_MID), (COL[0] + BW, LRN_MID))

txt("head.lrn", W / 2, MARGIN + HEAD_H / 2,
    "Learning: Recover the Hidden Urgency", FS_HEAD, "bold")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
EPS = 1e-7


def _o(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _between(a, b, c):
    return (min(a[0], b[0]) - EPS <= c[0] <= max(a[0], b[0]) + EPS and
            min(a[1], b[1]) - EPS <= c[1] <= max(a[1], b[1]) + EPS)


def seg_seg(p1, p2, p3, p4):
    d1, d2 = _o(p3, p4, p1), _o(p3, p4, p2)
    d3, d4 = _o(p1, p2, p3), _o(p1, p2, p4)
    if (((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and
            ((d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS))):
        return True
    if abs(d1) <= EPS and _between(p3, p4, p1):
        return True
    if abs(d2) <= EPS and _between(p3, p4, p2):
        return True
    if abs(d3) <= EPS and _between(p1, p2, p3):
        return True
    if abs(d4) <= EPS and _between(p1, p2, p4):
        return True
    return False


def in_rect(p, r):
    x, y, w, h = r
    return x - EPS <= p[0] <= x + w + EPS and y - EPS <= p[1] <= y + h + EPS


def seg_rect(p0, p1, r):
    """True if the closed segment meets the closed rectangle anywhere."""
    if in_rect(p0, r) or in_rect(p1, r):
        return True
    x, y, w, h = r
    c = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return any(seg_seg(p0, p1, c[i], c[(i + 1) % 4]) for i in range(4))


def rect_rect(a, b):
    return (a[0] < b[0] + b[2] - EPS and b[0] < a[0] + a[2] - EPS and
            a[1] < b[1] + b[3] - EPS and b[1] < a[1] + a[3] - EPS)


def verify():
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    tboxes = []
    for name, owner, t in TEXTS:
        bb = t.get_window_extent(rend)
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        tboxes.append((name, owner, (x0, y0, x1 - x0, y1 - y0)))

    fail = []
    # 1. arrow x arrow
    for i in range(len(SEGS)):
        for j in range(i + 1, len(SEGS)):
            n1, a0, a1 = SEGS[i]
            n2, b0, b1 = SEGS[j]
            if seg_seg(a0, a1, b0, b1):
                fail.append(f"arrow/arrow: {n1} x {n2}")
    # 2. arrow x box
    for n, a0, a1 in SEGS:
        for bn, bx, by, bw, bh in BOXES:
            if seg_rect(a0, a1, (bx, by, bw, bh)):
                fail.append(f"arrow/box: {n} x {bn}")
    # 3. text x arrow
    for tn, _, tr in tboxes:
        for n, a0, a1 in SEGS:
            if seg_rect(a0, a1, tr):
                fail.append(f"text/arrow: {tn} x {n}")
    # 4. text x foreign box, and own text inside its own box
    for tn, owner, tr in tboxes:
        for bn, bx, by, bw, bh in BOXES:
            r = (bx, by, bw, bh)
            if bn == owner:
                if not (tr[0] >= bx and tr[1] >= by and
                        tr[0] + tr[2] <= bx + bw and tr[1] + tr[3] <= by + bh):
                    fail.append(f"overflow: {tn} escapes {bn}")
            elif rect_rect(tr, r):
                fail.append(f"text/box: {tn} x {bn}")
    # 5. text x text
    for i in range(len(tboxes)):
        for j in range(i + 1, len(tboxes)):
            if rect_rect(tboxes[i][2], tboxes[j][2]):
                fail.append(f"text/text: {tboxes[i][0]} x {tboxes[j][0]}")
    # 6. everything on canvas
    for tn, _, tr in tboxes:
        if tr[0] < 0 or tr[1] < 0 or tr[0] + tr[2] > W or tr[1] + tr[3] > H:
            fail.append(f"off-canvas: {tn}")

    n_pairs = len(SEGS) * (len(SEGS) - 1) // 2
    print(f"[check] {len(SEGS)} arrows, {len(BOXES)} boxes, {len(tboxes)} text blocks")
    print(f"[check] arrow x arrow pairs tested : {n_pairs}")
    print(f"[check] arrow x box  pairs tested  : {len(SEGS) * len(BOXES)}")
    print(f"[check] text  x arrow pairs tested : {len(tboxes) * len(SEGS)}")
    print(f"[check] text  x box   pairs tested : {len(tboxes) * len(BOXES)}")
    print(f"[check] text  x text  pairs tested : {len(tboxes) * (len(tboxes) - 1) // 2}")
    if fail:
        print(f"[check] FAIL ({len(fail)}):")
        for f in fail:
            print("        " + f)
    else:
        print("[check] PASS: no arrow crosses an arrow, no arrow touches a box, "
              "no text overlaps a box, an arrow, or other text.")
    # tightest padding inside a stage box, and the closest text/ink gap
    side, top, bot = [], [], []
    for tn, owner, tr in tboxes:
        if owner is None:
            continue
        bx, by, bw, bh = next(b[1:] for b in BOXES if b[0] == owner)
        side.append((min(tr[0] - bx, bx + bw - tr[0] - tr[2]), tn))
        if tn.endswith(".title"):
            top.append((by + bh - tr[1] - tr[3], tn))
        else:
            bot.append((tr[1] - by, tn))
    print(f"[check] tightest padding inside a box: side {min(side)[0]:.3f} cm "
          f"({min(side)[1]}), top {min(top)[0]:.3f} cm, bottom {min(bot)[0]:.3f} cm")
    gaps = []
    for tn, _, tr in tboxes:
        for n, a0, a1 in SEGS:
            if a0[0] == a1[0]:                     # vertical arrow
                d = max(a0[0] - (tr[0] + tr[2]), tr[0] - a0[0])
                lo, hi = sorted((a0[1], a1[1]))
                if tr[1] < hi and tr[1] + tr[3] > lo:
                    gaps.append((d, f"{tn} | {n}"))
    print(f"[check] tightest text-to-arrow gap: {min(gaps)[0]:.3f} cm "
          f"({min(gaps)[1]})")
    wid = max(tboxes, key=lambda r: r[2][2])
    print(f"[check] widest text block: {wid[0]} = {wid[2][2]:.2f} cm")
    return not fail


ok = verify()
fig.savefig(OUT, format="pdf")
print(f"wrote {OUT}  ({W:.2f} x {H:.2f} cm)")
if len(sys.argv) > 1:
    fig.savefig(sys.argv[1], format="png", dpi=300)
    print(f"wrote {sys.argv[1]}")
sys.exit(0 if ok else 1)
