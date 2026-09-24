"""
Recreation of the schematic

    "3) From hidden-representation overlaps to the superposition matrix"

Pure matplotlib -- no LaTeX installation required.
Everything is drawn into one axes using a 150 x 100 "canvas" coordinate system
(figure is 15 x 10 in, so one canvas unit = 0.1 in in both directions).
"""

import os

import matplotlib
OUT_DIR = os.path.dirname(os.path.abspath(__file__))  # figures are saved next to this script
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, PathPatch
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox

# ----------------------------------------------------------------------
# The two colorbars are redundant -- switch them on/off here.
# ----------------------------------------------------------------------
SHOW_VERTICAL_COLORBAR   = False     # small one next to the matrix C
SHOW_HORIZONTAL_COLORBAR = False    # one under the overlap map  -> left out

# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",        # Computer-Modern-like math, as in the original
})

BLUE   = "#2432d4"      # blue math annotations
ORANGE = "#f39200"      # highlight colour of the reference pixel j
ORANGE2 = "#e8590c"     # slightly darker orange used for c^(j) = C_:j
BOX_FC = "#fdf5e3"      # caption-box fill
BOX_EC = "#e9a53c"      # caption-box edge
FRAME  = "#c3d2f2"      # outer frame

CMAP = LinearSegmentedColormap.from_list("bluewhitered",
                                         ["#1b4aa0", "#ffffff", "#c62222"])
NORM = Normalize(vmin=-1, vmax=1)

FS_HEAD, FS_MATH, FS_CAP = 14, 20, 12.5

# ----------------------------------------------------------------------
# Data of the two heatmaps
# ----------------------------------------------------------------------
C = np.array([
    [ 1.00,  0.07, -0.22,  0.09,  0.06, -0.16],
    [ 0.07,  1.00,  0.17,  0.15, -0.18,  0.09],
    [-0.22,  0.17,  1.00,  0.34,  0.07, -0.12],
    [ 0.09,  0.15,  0.34,  1.00,  0.14,  0.06],
    [ 0.06, -0.18,  0.07,  0.14,  1.00, -0.15],
    [-0.16,  0.09, -0.12,  0.06, -0.15,  1.00],
])

OVERLAP = np.array([
    [ 0.55,  0.18, -0.05,  0.20,  0.50],
    [ 0.20, -0.45,  0.05,  0.08,  0.28],
    [-0.02,  0.25,  0.00,  0.30,  0.55],
    [ 0.05,  0.30,  0.18, -0.62, -0.40],
    [-0.05,  0.25,  0.45, -0.18,  0.05],
])

J_ROW, J_COL = 2, 1        # reference pixel in the 5 x 5 image (0-based)
J_IDX        = 3           # highlighted column of C            (0-based)

# ----------------------------------------------------------------------
# Canvas
# ----------------------------------------------------------------------
FIG_W_PX, FIG_H_PX = 1518, 1032          # pixel format of the original figure
FIG_W_IN = 15.0                          # canvas: 1 unit = 0.1 in in x and y
DPI      = FIG_W_PX / FIG_W_IN           # -> 101.2
FIG_H_IN = FIG_H_PX / DPI
SCALE    = 1                             # 2 -> same format, twice the pixels

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
H_UNITS = 150.0 * FIG_H_PX / FIG_W_PX    # canvas height matching the aspect
PAD     = (H_UNITS - 100.0) / 2.0        # keep the old 150 x 100 layout centred
ax.set_xlim(0, 150)
ax.set_ylim(-PAD, 100 + PAD)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)

ax.add_patch(FancyBboxPatch((2, 2), 146, 96,
                            boxstyle="round,pad=0,rounding_size=1.6",
                            facecolor="white", edgecolor=FRAME, linewidth=1.6))

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def heatmap(x0, y_top, cell, data, frame=False):
    nrow, ncol = data.shape
    for r in range(nrow):
        for c in range(ncol):
            ax.add_patch(Rectangle((x0 + c * cell, y_top - (r + 1) * cell),
                                   cell, cell,
                                   facecolor=CMAP(NORM(data[r, c])),
                                   edgecolor="none"))
    if frame:
        ax.add_patch(Rectangle((x0, y_top - nrow * cell), ncol * cell,
                               nrow * cell, facecolor="none",
                               edgecolor="black", linewidth=1.4))


def empty_grid(x0, y_top, cell, n=5):
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((x0 + c * cell, y_top - (r + 1) * cell),
                                   cell, cell, facecolor="white",
                                   edgecolor="black", linewidth=1.2))


def highlight(x, y, w, h, lw=2.2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=ORANGE, alpha=0.22,
                           edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=ORANGE,
                           linewidth=lw, zorder=4))


def bracket(xc, half_w, y_bot, y_top, lw=1.4, tick=0.9):
    """Tall square brackets [ ] centred at xc."""
    for x, s in ((xc - half_w, +1), (xc + half_w, -1)):
        ax.plot([x, x], [y_bot, y_top], color="black", lw=lw,
                solid_capstyle="butt")
        ax.plot([x, x + s * tick], [y_top, y_top], color="black", lw=lw)
        ax.plot([x, x + s * tick], [y_bot, y_bot], color="black", lw=lw)


def arrow(x0, x1, y, lw=2.4):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=lw,
                                mutation_scale=22, shrinkA=0, shrinkB=0))


def curly_brace(x1, x2, y, depth=2.6, color=BLUE, lw=1.5):
    """Downward-pointing curly brace spanning [x1, x2] at height y."""
    m, h = (x1 + x2) / 2.0, (x2 - x1) / 2.0
    verts = [(x1, y),
             (x1 + 0.12 * h, y - depth), (m - 0.55 * h, y - 0.12 * depth), (m, y - depth),
             (m + 0.55 * h, y - 0.12 * depth), (x2 - 0.12 * h, y - depth), (x2, y)]
    codes = [Path.MOVETO] + [Path.CURVE4] * 6
    ax.add_patch(PathPatch(Path(verts, codes), facecolor="none",
                           edgecolor=color, lw=lw))


def rich_text(x, y, lines, fs=FS_CAP, sep_line=5.0, weights=None):
    """Centred multi-line text whose segments may have individual colours.

    `lines` is a list of lines; every line is a list of (string, colour) pairs.
    """
    rows = []
    for line in lines:
        parts = [TextArea(txt, textprops=dict(color=col, fontsize=fs,
                                              fontweight=(weights or "normal")))
                 for txt, col in line]
        rows.append(HPacker(children=parts, align="baseline", pad=0, sep=0))
    box = VPacker(children=rows, align="center", pad=0, sep=sep_line)
    ax.add_artist(AnnotationBbox(box, (x, y), frameon=False,
                                 box_alignment=(0.5, 0.5), pad=0))


def head_math(x, y, head, math, fs_head=FS_HEAD, fs_math=FS_MATH):
    """Bold heading followed by an italic math symbol, centred at (x, y)."""
    parts = [TextArea(head, textprops=dict(fontsize=fs_head, fontweight="bold")),
             TextArea(math, textprops=dict(fontsize=fs_math))]
    ax.add_artist(AnnotationBbox(HPacker(children=parts, align="center",
                                         pad=0, sep=4),
                                 (x, y), frameon=False,
                                 box_alignment=(0.5, 0.5), pad=0))


def caption(x0, x1, y0, y1, lines, fs=FS_CAP):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=BOX_FC, edgecolor=BOX_EC,
                                linewidth=1.3))
    rich_text((x0 + x1) / 2, (y0 + y1) / 2, lines, fs=fs)


# ----------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------
ax.text(5, 93.5,
        "3) From hidden-representation overlaps to the superposition matrix",
        fontsize=19, fontweight="bold", ha="left", va="center")

Y_MID    = 55.0      # vertical centre of the pipeline
Y_HEAD   = 85.0      # bold headings
Y_FORM   = 79.0      # blue formulas

# ----------------------------------------------------------------------
# 1) Input image (5 x 5)
# ----------------------------------------------------------------------
IMG_X0, IMG_CELL = 2.5, 4.2
IMG_TOP = Y_MID + 2.5 * IMG_CELL
empty_grid(IMG_X0, IMG_TOP, IMG_CELL)

hx = IMG_X0 + J_COL * IMG_CELL
hy = IMG_TOP - (J_ROW + 1) * IMG_CELL
highlight(hx, hy, IMG_CELL, IMG_CELL)
ax.text(hx + IMG_CELL / 2, hy + IMG_CELL / 2, "$j$", color=ORANGE,
        fontsize=15, ha="center", va="center", zorder=5)

head_math(IMG_X0 + 2.5 * IMG_CELL, 82.0, "Input image", r"$(5 \times 5)$")

arrow(24.9, 27.6, Y_MID)   # input grid ends at 23.5, brackets start at 28.2

# ----------------------------------------------------------------------
# 2) Weight vectors
# ----------------------------------------------------------------------
VEC_TOP, VEC_BOT = Y_MID + 11.0, Y_MID - 11.0
xs     = [30.5, 37.5, 50.5, 63.5]
labels = ["$w_1$", "$w_2$", "$w_j$", "$w_d$"]
cols   = ["black", "black", ORANGE, "black"]
for x, lab, col in zip(xs, labels, cols):
    bracket(x, 2.3, VEC_BOT, VEC_TOP)
    ax.text(x, Y_MID, r"$\vdots$", fontsize=20, ha="center", va="center")
    ax.text(x, VEC_TOP + 3.4, lab, fontsize=20, color=col,
            ha="center", va="center")
for x in (44.0, 57.0):
    ax.text(x, Y_MID, r"$\cdots$", fontsize=20, ha="center", va="center")

VEC_CX = 47.0
ax.text(VEC_CX, Y_HEAD, "Weight vectors", fontsize=FS_HEAD,
        fontweight="bold", ha="center", va="center")
ax.text(VEC_CX, Y_FORM, r"$w_i \in \mathbb{R}^n$", fontsize=FS_MATH,
        color=BLUE, ha="center", va="center")

curly_brace(28.0, 66.0, VEC_BOT - 2.5, depth=2.6)
ax.text(VEC_CX, 36.5,
        r"$W = [w_1, w_2, \ldots, w_d] \in \mathbb{R}^{n \times d}$",
        fontsize=FS_MATH, color=BLUE, ha="center", va="center")

arrow(66.6, 68.9, Y_MID)   # w_d bracket ends at 65.8, row labels start at ~69.7

# ----------------------------------------------------------------------
# 3) Superposition matrix C
# ----------------------------------------------------------------------
MAT_X0, MAT_CELL = 73.0, 3.6
MAT_TOP = Y_MID + 3 * MAT_CELL
MAT_CX  = MAT_X0 + 3 * MAT_CELL
heatmap(MAT_X0, MAT_TOP, MAT_CELL, C, frame=True)
highlight(MAT_X0 + J_IDX * MAT_CELL, MAT_TOP - 6 * MAT_CELL - 1.1,
          MAT_CELL, 6 * MAT_CELL + 2.2, lw=2.0)

col_lab = ["$1$", "$2$", r"$\cdots$", "$j$", r"$\cdots$", "$d$"]
row_lab = ["$1$", "$2$", r"$\vdots$", "$i$", r"$\vdots$", "$d$"]
for k in range(6):
    ax.text(MAT_X0 + (k + 0.5) * MAT_CELL, MAT_TOP + 2.8, col_lab[k],
            fontsize=14, ha="center", va="center",
            color=ORANGE if k == J_IDX else "black")
    ax.text(MAT_X0 - 2.6, MAT_TOP - (k + 0.5) * MAT_CELL, row_lab[k],
            fontsize=14, ha="center", va="center")

head_math(MAT_CX, Y_HEAD, "Superposition matrix", "$C$")
ax.text(MAT_CX, Y_FORM - 0.5, r"$C_{ij} = \dfrac{1}{n}\, w_i^{\,T} w_j$",
        fontsize=FS_MATH, color=BLUE, ha="center", va="center")
ax.text(MAT_CX, 39.0, r"$d \times d$", fontsize=FS_MATH,
        ha="center", va="center")

# ---- small vertical colorbar next to C -------------------------------
if SHOW_VERTICAL_COLORBAR:
    cb_x, cb_w, cb_y0, cb_h = 96.8, 2.2, 46.0, 18.0
    ax.imshow(np.linspace(1, -1, 256).reshape(-1, 1), cmap=CMAP, norm=NORM,
              aspect="auto", extent=[cb_x, cb_x + cb_w, cb_y0, cb_y0 + cb_h],
              zorder=2)
    ax.add_patch(Rectangle((cb_x, cb_y0), cb_w, cb_h, facecolor="none",
                           edgecolor="black", lw=0.8, zorder=3))
    for frac, lab in ((1.0, "$1$"), (0.5, "$0$"), (0.0, "$-1$")):
        ax.text(cb_x + cb_w + 1.0, cb_y0 + frac * cb_h, lab, fontsize=13,
                ha="left", va="center")

A3_X0 = 102.3 if SHOW_VERTICAL_COLORBAR else 96.2
arrow(A3_X0, 105.8, Y_MID)  # column-vector bracket starts at 107.5

# ----------------------------------------------------------------------
# 4) Column j of C
# ----------------------------------------------------------------------
VEC2_X, VEC2_TOP, VEC2_BOT = 112.5, Y_MID + 15.0, Y_MID - 15.0
bracket(VEC2_X, 5.0, VEC2_BOT, VEC2_TOP, tick=1.4)
entries = ["$C_{1j}$", "$C_{2j}$", r"$\vdots$", "$C_{ij}$", r"$\vdots$", "$C_{dj}$"]
for y, e in zip(np.linspace(VEC2_TOP - 2.8, VEC2_BOT + 2.8, 6), entries):
    ax.text(VEC2_X, y, e, fontsize=14, ha="center", va="center")

head_math(VEC2_X, Y_HEAD, "column", "$j$")
ax.text(VEC2_X, Y_FORM, "$c^{(j)} = C_{:j}$", fontsize=FS_MATH, color=ORANGE2,
        ha="center", va="center")
ax.text(VEC2_X, 73.0, "$c^{(j)} =$", fontsize=FS_MATH, color=BLUE,
        ha="center", va="center")
ax.text(VEC2_X, 34.0, r"$\in \mathbb{R}^d$", fontsize=FS_MATH, color=BLUE,
        ha="center", va="center")

arrow(118.9, 122.2, Y_MID)  # bracket ends at 117.5, overlap map starts at 123.5

# ----------------------------------------------------------------------
# 5) Overlap map in the original input geometry
# ----------------------------------------------------------------------
MAP_CELL = 3.8
MAP_CX   = 133.0
MAP_X0   = MAP_CX - 2.5 * MAP_CELL
MAP_TOP  = Y_MID + 2.5 * MAP_CELL
heatmap(MAP_X0, MAP_TOP, MAP_CELL, OVERLAP)

hx = MAP_X0 + J_COL * MAP_CELL
hy = MAP_TOP - (J_ROW + 1) * MAP_CELL
highlight(hx, hy, MAP_CELL, MAP_CELL)
ax.text(hx + MAP_CELL / 2, hy + MAP_CELL / 2, "$j$", color=ORANGE,
        fontsize=15, ha="center", va="center", zorder=5)

ax.text(MAP_CX, Y_HEAD + 1.5, "Overlap map in", fontsize=13.5,
        fontweight="bold", ha="center", va="center")
ax.text(MAP_CX, Y_HEAD - 3.5, "original input geometry", fontsize=13.5,
        fontweight="bold", ha="center", va="center")
ax.text(MAP_CX, 76.0, r"$(5 \times 5)$", fontsize=FS_HEAD,
        ha="center", va="center")

if SHOW_HORIZONTAL_COLORBAR:
    cb_x0, cb_w, cb_y, cb_h = MAP_CX - 9.0, 18.0, 39.5, 2.2
    ax.imshow(np.linspace(-1, 1, 256).reshape(1, -1), cmap=CMAP, norm=NORM,
              aspect="auto", extent=[cb_x0, cb_x0 + cb_w, cb_y, cb_y + cb_h],
              zorder=2)
    ax.add_patch(Rectangle((cb_x0, cb_y), cb_w, cb_h, facecolor="none",
                           edgecolor="black", lw=0.8, zorder=3))
    for frac, lab in ((0.0, "$-1$"), (0.5, "$0$"), (1.0, "$1$")):
        ax.text(cb_x0 + frac * cb_w, cb_y - 2.6, lab, fontsize=13,
                ha="center", va="top")

# ----------------------------------------------------------------------
# 6) Caption boxes
# ----------------------------------------------------------------------
K = "black"
caption(3.5, 32.0, 12.5, 26.0,
        [[("Each pixel ", K), ("$i$", ORANGE2), (" corresponds", K)],
         [("to a weight vector $w_i$", K)],
         [("(column in $W$).", K)]])

caption(37.0, 79.0, 15.0, 26.0,
        [[("$C_{ij}$ measures the normalized overlap", K)],
         [("between weight vectors $w_i$ and $w_j$.", K)]])

caption(83.0, 114.0, 15.0, 26.0,
        [[("Column ", K), ("$j$", ORANGE2), (" contains the overlap", K)],
         [("of $w_j$ with all other $w_i$.", K)]])

caption(117.0, 146.5, 9.5, 26.0,
        [[("Pixel $i$ shows $C_{ij}$:", K)],
         [("the overlap between pixel $i$", K)],
         [("and the reference pixel $j$", K)],
         [("in hidden-representation space.", K)]])

fig.savefig(os.path.join(OUT_DIR, "from_hidden_rep_to_superposition.png"),
            dpi=DPI * SCALE, facecolor="white")

print("saved")
