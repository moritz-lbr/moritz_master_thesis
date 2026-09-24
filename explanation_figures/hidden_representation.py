"""
Recreation of the schematic

    "1) Mapping a single input pixel to the hidden layer"

Pure matplotlib -- no LaTeX installation required.
Canvas: 1 unit = 0.1 in in x and y; the output PNG has exactly the format of
the original figure (1672 x 941 px).
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

# ----------------------------------------------------------------------
# Style / colours
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",
})

ORANGE   = "#f28c00"      # highlights and orange annotations
ORANGE_F = "#fdf1d8"      # light orange fill
BLUE     = "#1f4fd8"      # blue math annotations
FRAME    = "#c3d2f2"      # outer frame
BOX_FC   = "#eef4fc"      # caption box fill
BOX_EC   = "#b9d0ee"      # caption box edge

FS_TITLE, FS_HEAD, FS_TXT, FS_MATH = 23, 15, 15, 20

# ----------------------------------------------------------------------
# Canvas -- exact format of the original figure
# ----------------------------------------------------------------------
FIG_W_PX, FIG_H_PX = 1672, 941
FIG_W_IN = 17.768                       # 1 canvas unit = 0.1 in
DPI      = FIG_W_PX / FIG_W_IN          # -> 94.1
FIG_H_IN = FIG_H_PX / DPI
SCALE    = 1                            # 2 -> same format, twice the pixels

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
W_UNITS = FIG_W_IN * 10.0               # 177.68
H_UNITS = FIG_H_IN * 10.0               # 100.0
ax.set_xlim(0, W_UNITS)
ax.set_ylim(0, H_UNITS)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)

ax.add_patch(FancyBboxPatch((1.5, 1.2), W_UNITS - 3.0, H_UNITS - 2.4,
                            boxstyle="round,pad=0,rounding_size=1.6",
                            facecolor="white", edgecolor=FRAME, linewidth=1.6))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def empty_grid(x0, y_top, cell, n=5, lw=1.2):
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((x0 + c * cell, y_top - (r + 1) * cell),
                                   cell, cell, facecolor="white",
                                   edgecolor="black", linewidth=lw))


def highlight(x, y, w, h, lw=2.4, color=ORANGE, fill=ORANGE_F):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor="none",
                           zorder=3))
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=color,
                           linewidth=lw, zorder=4))


def bracket(xc, half_w, y_bot, y_top, lw=1.5, tick=1.5):
    """Tall square brackets [ ] centred at xc."""
    for x, s in ((xc - half_w, +1), (xc + half_w, -1)):
        ax.plot([x, x], [y_bot, y_top], color="black", lw=lw,
                solid_capstyle="butt", zorder=5)
        ax.plot([x, x + s * tick], [y_top, y_top], color="black", lw=lw, zorder=5)
        ax.plot([x, x + s * tick], [y_bot, y_bot], color="black", lw=lw, zorder=5)


def arrow(x0, y0, x1, y1, lw=2.4, color="black", scale=22):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=scale, shrinkA=0, shrinkB=0))


def head_math(x, y, head, math, fs_head=FS_HEAD, fs_math=FS_MATH,
              color="black", weight="bold", sep=4):
    """Bold heading followed by a math symbol, centred at (x, y)."""
    parts = [TextArea(head, textprops=dict(fontsize=fs_head, fontweight=weight,
                                           color=color)),
             TextArea(math, textprops=dict(fontsize=fs_math, color=color))]
    ax.add_artist(AnnotationBbox(HPacker(children=parts, align="center",
                                         pad=0, sep=sep),
                                 (x, y), frameon=False,
                                 box_alignment=(0.5, 0.5), pad=0))


# ----------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------
ax.text(4.5, 94.0, "1) Mapping a single input pixel to the hidden layer",
        fontsize=FS_TITLE, fontweight="bold", ha="left", va="center")

Y_MID = 58.9        # vertical centre of the pipeline

# ----------------------------------------------------------------------
# 1) Input image
# ----------------------------------------------------------------------
IMG_X0, IMG_CELL = 8.0, 6.3
IMG_TOP = 75.6
IMG_CX  = IMG_X0 + 2.5 * IMG_CELL
empty_grid(IMG_X0, IMG_TOP, IMG_CELL)

hx = IMG_X0 + 2 * IMG_CELL                    # selected pixel: centre cell
hy = IMG_TOP - 3 * IMG_CELL
highlight(hx, hy, IMG_CELL, IMG_CELL)

head_math(IMG_CX, 79.8, "Input image", r"$(5 \times 5)$")
head_math(IMG_CX, 37.0, "Selected pixel", "$(r, c)$", color=ORANGE)
ax.text(IMG_CX, 32.0, r"(value $= x_i$)", fontsize=FS_TXT, color=ORANGE,
        fontweight="bold", ha="center", va="center")

# ----------------------------------------------------------------------
# 2) Flatten arrow
# ----------------------------------------------------------------------
arrow(42.5, Y_MID, 52.5, Y_MID)
ax.text(47.5, 63.8, "Flatten", fontsize=FS_TXT, ha="center", va="center")

# ----------------------------------------------------------------------
# 3) Flattened input vector x
# ----------------------------------------------------------------------
XV_CX = 68.5
bracket(XV_CX, 4.8, 40.7, 75.8)
x_entries = ["$x_1$", "$x_2$", r"$\vdots$", "$x_i$", r"$\vdots$", "$x_d$"]
x_ys      = [73.1, 67.9, 62.2, 55.4, 49.0, 43.4]
highlight(XV_CX - 3.6, x_ys[3] - 2.4, 7.2, 4.8, lw=2.0)
for y, e in zip(x_ys, x_entries):
    ax.text(XV_CX, y, e, fontsize=20, ha="center", va="center", zorder=6)

head_math(XV_CX, 85.0, "Flattened input vector", r"$\boldsymbol{x}$")
ax.text(XV_CX, 80.3, "$(d = 25)$", fontsize=FS_MATH, ha="center", va="center")
ax.text(XV_CX, 35.0, r"$\boldsymbol{x} \in \mathbb{R}^d$", fontsize=17, color=BLUE,
        ha="center", va="center")

# "Same pixel" annotation
ax.text(51.5, 51.8, "Same", fontsize=FS_TXT, color=ORANGE, fontweight="bold",
        ha="center", va="center")
ax.text(51.5, 47.2, "pixel", fontsize=FS_TXT, color=ORANGE, fontweight="bold",
        ha="center", va="center")
arrow(56.5, 50.3, 63.9, 55.6, lw=2.0, color=ORANGE, scale=18)

# ----------------------------------------------------------------------
# 4) Weight matrix W
# ----------------------------------------------------------------------
arrow(75.5, Y_MID, 82.5, Y_MID)

W_XS = [95.1, 104.7, 114.2, 121.7, 130.7, 138.7]     # column centres
W_YS = [68.9, 62.8, 54.3, 46.3]                      # row centres
W_CX = 0.5 * (W_XS[0] + W_XS[-1])

bracket(W_CX, 28.6, 42.5, 72.5)

# highlighted column w_i
highlight(W_XS[3] - 3.7, 43.1, 7.4, 29.1, lw=2.0)

col_head = [r"$\boldsymbol{w}_1$", r"$\boldsymbol{w}_2$", r"$\cdots$",
            r"$\boldsymbol{w}_i$", r"$\cdots$", r"$\boldsymbol{w}_d$"]
for x, h in zip(W_XS, col_head):
    ax.text(x, 74.8, h, fontsize=20, ha="center", va="center")

row_head = ["$1$", "$2$", r"$\vdots$", "$n$"]
for y, h in zip(W_YS, row_head):
    ax.text(86.1, y, h, fontsize=20, ha="center", va="center")

rows = [["$w_{1,1}$", "$w_{1,2}$", r"$\cdots$", "$w_{1,j}$", r"$\cdots$", "$w_{1,d}$"],
        ["$w_{2,1}$", "$w_{2,2}$", r"$\cdots$", "$w_{2,j}$", r"$\cdots$", "$w_{2,d}$"],
        [r"$\vdots$",  r"$\vdots$", r"$\ddots$", r"$\vdots$", r"$\ddots$", r"$\vdots$"],
        ["$w_{n,1}$", "$w_{n,2}$", r"$\cdots$", "$w_{n,j}$", r"$\cdots$", "$w_{n,d}$"]]
for y, row in zip(W_YS, rows):
    for x, e in zip(W_XS, row):
        ax.text(x, y, e, fontsize=20, ha="center", va="center", zorder=6)

head_math(W_CX, 85.9, "Weight matrix", r"$\boldsymbol{W}$")
ax.text(W_CX, 81.1, r"$(n \times d)$", fontsize=FS_MATH, ha="center", va="center")

for y, line in zip([38.9, 34.4, 30.1],
                   [r"Column $\boldsymbol{w}_i$ corresponds to the",
                    r"connections from input pixel $i$",
                    r"to all hidden units"]):
    ax.text(113.2, y, line, fontsize=FS_TXT, color=ORANGE, fontweight="bold",
            ha="center", va="center")

# ----------------------------------------------------------------------
# 5) Pre-activation h(x)
# ----------------------------------------------------------------------
arrow(147.5, Y_MID, 154.5, Y_MID)

HV_CX = 161.5
bracket(HV_CX, 5.6, 42.5, 72.5)
for y, e in zip([68.9, 62.8, 54.3, 46.3],
                ["$h_1(x)$", "$h_2(x)$", r"$\vdots$", "$h_n(x)$"]):
    ax.text(HV_CX, y, e, fontsize=20, ha="center", va="center")

ax.text(HV_CX, 86.5, "Pre-activation", fontsize=FS_HEAD, fontweight="bold",
        ha="center", va="center")
ax.text(HV_CX, 82.8, "(hidden)", fontsize=FS_HEAD, fontweight="bold",
        ha="center", va="center")
ax.text(HV_CX, 78.8, r"$\boldsymbol{h}(\boldsymbol{x}) = \boldsymbol{W}\boldsymbol{x}$",
        fontsize=17, ha="center", va="center")
ax.text(HV_CX, 36.0, r"$\boldsymbol{h}(\boldsymbol{x}) \in \mathbb{R}^n$",
        fontsize=17, color=BLUE, ha="center", va="center")

# ----------------------------------------------------------------------
# 6) Caption box
# ----------------------------------------------------------------------
ax.add_patch(FancyBboxPatch((12.8, 10.7), 156.2, 9.1,
                            boxstyle="round,pad=0,rounding_size=1.2",
                            facecolor=BOX_FC, edgecolor=BOX_EC, linewidth=1.3))
ax.text(91.0, 15.3,
        r"The column $\boldsymbol{w}_i$ of $\boldsymbol{W}$ collects how input pixel $i$ "
        r"contributes (with possibly different strengths and signs) to "
        r"$\mathit{every}$ hidden unit.",
        fontsize=18, ha="center", va="center")

fig.savefig("explanation_figures/hidden_representations.png",
            dpi=DPI * SCALE, facecolor="white")

print("saved")
