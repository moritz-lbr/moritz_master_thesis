"""
Recreation of the schematic

    "2) How hidden representations from different input pixels can overlap
        or be orthogonal"

Pure matplotlib -- no LaTeX installation required.
Canvas: 1 unit = 0.1 in in x and y; the output PNG has exactly the format of
the original figure (2149 x 731 px).
"""

import os

import matplotlib
OUT_DIR = os.path.dirname(os.path.abspath(__file__))  # figures are saved next to this script
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox

# ----------------------------------------------------------------------
# Style / colours
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",
})

RED, GREEN, BLUE = "#e8000d", "#00963f", "#0b3fd4"
RED_F, GREEN_F, BLUE_F = "#fdecec", "#eafaee", "#eaeffc"
FRAME  = "#c3d2f2"
YEL_FC, YEL_EC = "#fdf3d3", "#f0b040"

FS_TITLE, FS_HEAD, FS_CASE, FS_TXT, FS_VEC, FS_FORM = 34, 19, 20, 17, 24, 30

# ----------------------------------------------------------------------
# Canvas -- exact format of the original figure
# ----------------------------------------------------------------------
FIG_W_PX, FIG_H_PX = 2149, 731
FIG_W_IN = 29.4                          # 1 canvas unit = 0.1 in
DPI      = FIG_W_PX / FIG_W_IN           # -> 73.1
FIG_H_IN = FIG_H_PX / DPI
SCALE    = 1                             # 2 -> same format, twice the pixels

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
W_UNITS, H_UNITS = FIG_W_IN * 10.0, FIG_H_IN * 10.0     # 294 x 100
ax.set_xlim(0, W_UNITS)
ax.set_ylim(0, H_UNITS)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)

ax.add_patch(FancyBboxPatch((1.4, 1.2), W_UNITS - 2.8, H_UNITS - 2.4,
                            boxstyle="round,pad=0,rounding_size=2.0",
                            facecolor="white", edgecolor=FRAME, linewidth=1.8))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def empty_grid(x0, y_top, cell, n=5, lw=1.4):
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((x0 + c * cell, y_top - (r + 1) * cell),
                                   cell, cell, facecolor="white",
                                   edgecolor="black", linewidth=lw))


def highlight_cell(x0, y_top, cell, row, col, color, fill, lw=3.0):
    ax.add_patch(Rectangle((x0 + col * cell, y_top - (row + 1) * cell),
                           cell, cell, facecolor=fill, edgecolor=color,
                           linewidth=lw, zorder=3))


def strip(x0, y_top, w, cell, color, n_top=4, n_bot=4, gap=6.6, lw=1.6):
    """Vertical stack of cells with a vertical-dots gap in the middle."""
    y = y_top
    for _ in range(n_top):
        ax.add_patch(Rectangle((x0, y - cell), w, cell, facecolor="white",
                               edgecolor=color, linewidth=lw))
        y -= cell
    ax.text(x0 + w / 2, y - gap / 2, r"$\vdots$", fontsize=20,
            ha="center", va="center")
    y -= gap
    for _ in range(n_bot):
        ax.add_patch(Rectangle((x0, y - cell), w, cell, facecolor="white",
                               edgecolor=color, linewidth=lw))
        y -= cell
    return y                                    # bottom edge


def arrow(x0, y0, x1, y1, color="black", lw=2.6, scale=26):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=scale, shrinkA=0, shrinkB=0))


def rich_text(x, y, lines, fs=FS_TXT, sep_line=6.0, align="center",
              box_alignment=(0.5, 0.5)):
    """Multi-line text whose segments can have individual colour/weight.

    `lines` is a list of lines; each line is a list of (text, colour, weight).
    """
    rows = []
    for line in lines:
        parts = [TextArea(t, textprops=dict(color=c, fontweight=w, fontsize=fs))
                 for (t, c, w) in line]
        rows.append(HPacker(children=parts, align="baseline", pad=0, sep=0))
    box = VPacker(children=rows, align=align, pad=0, sep=sep_line)
    ax.add_artist(AnnotationBbox(box, (x, y), frameon=False,
                                 box_alignment=box_alignment, pad=0))


def axes_triad(ox, oy, up, right, diag, lw=2.2):
    """The little 'pseudo-3D' coordinate cross used in both panels."""
    arrow(ox, oy, ox, oy + up, lw=lw, scale=24)
    arrow(ox, oy, ox + right, oy, lw=lw, scale=24)
    arrow(ox, oy, ox - diag[0], oy - diag[1], lw=lw, scale=24)
    ax.plot([ox], [oy], "o", color="black", markersize=11, zorder=5)


# ----------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------
ax.text(5.5, 92.5,
        "2) How hidden representations from different input pixels "
        "can overlap or be orthogonal",
        fontsize=FS_TITLE, fontweight="bold", ha="left", va="center")

# ----------------------------------------------------------------------
# 1) Input image with three picked pixels
# ----------------------------------------------------------------------
IMG_X0, IMG_TOP, IMG_CELL = 8.2, 72.6, 8.7
IMG_CX = IMG_X0 + 2.5 * IMG_CELL
empty_grid(IMG_X0, IMG_TOP, IMG_CELL)
highlight_cell(IMG_X0, IMG_TOP, IMG_CELL, 0, 2, RED,   RED_F)
highlight_cell(IMG_X0, IMG_TOP, IMG_CELL, 2, 2, GREEN, GREEN_F)
highlight_cell(IMG_X0, IMG_TOP, IMG_CELL, 4, 1, BLUE,  BLUE_F)

ax.text(IMG_CX, 78.5, "Pick three input pixels", fontsize=FS_HEAD,
        fontweight="bold", ha="center", va="center")

# ----------------------------------------------------------------------
# 2) The corresponding columns of W
# ----------------------------------------------------------------------
STRIP_TOP, STRIP_W, STRIP_CELL = 66.5, 5.1, 4.4
for x0, color, lab in ((60.6, RED,   r"$\boldsymbol{w_p}$"),
                       (75.2, GREEN, r"$\boldsymbol{w_q}$"),
                       (89.6, BLUE,  r"$\boldsymbol{w_r}$")):
    strip(x0, STRIP_TOP, STRIP_W, STRIP_CELL, color)
    ax.text(x0 + STRIP_W / 2, 69.8, lab, fontsize=FS_VEC, color=color,
            ha="center", va="center")

ax.text(77.6, 81.9, "Their corresponding", fontsize=FS_HEAD,
        fontweight="bold", ha="center", va="center")
parts = [TextArea("columns in ", textprops=dict(fontsize=FS_HEAD, fontweight="bold")),
         TextArea(r"$W$", textprops=dict(fontsize=FS_HEAD + 2))]
ax.add_artist(AnnotationBbox(HPacker(children=parts, align="center", pad=0, sep=0),
                             (77.6, 76.5), frameon=False,
                             box_alignment=(0.5, 0.5), pad=0))

arrow(99.0, 48.0, 104.0, 48.0, lw=3.0, scale=24)

# ----------------------------------------------------------------------
# 3) Panel with case A and case B
# ----------------------------------------------------------------------
PANEL_X0, PANEL_X1, PANEL_Y0, PANEL_Y1 = 106.0, 215.0, 7.0, 88.0
ax.add_patch(FancyBboxPatch((PANEL_X0, PANEL_Y0), PANEL_X1 - PANEL_X0,
                            PANEL_Y1 - PANEL_Y0,
                            boxstyle="round,pad=0,rounding_size=2.0",
                            facecolor="white", edgecolor=FRAME, linewidth=1.8))
ax.plot([162.0, 162.0], [11.0, 84.5], color=FRAME, lw=1.5)

# ---- Case A: non-zero overlap ----------------------------------------
ax.text(133.0, 82.5, "Case A: Non-zero overlap", fontsize=FS_CASE,
        fontweight="bold", color=RED, ha="center", va="center")
ax.text(133.0, 77.0, "(not orthogonal)", fontsize=FS_CASE, fontweight="bold",
        color=RED, ha="center", va="center")

OAx, OAy = 116.3, 41.9
axes_triad(OAx, OAy, up=30.6, right=39.2, diag=(7.3, 6.9))
for tip, color, lab, off in (((130.6, 63.0), RED,   r"$\boldsymbol{w_p}$", (1.2, 2.6)),
                             ((136.8, 60.3), GREEN, r"$\boldsymbol{w_q}$", (2.6, 1.8)),
                             ((145.7, 56.2), BLUE,  r"$\boldsymbol{w_r}$", (2.8, 1.6))):
    arrow(OAx, OAy, *tip, color=color, lw=2.6, scale=22)
    ax.text(tip[0] + off[0], tip[1] + off[1], lab, fontsize=FS_VEC,
            color=color, ha="center", va="center")

ax.add_patch(FancyBboxPatch((107.4, 9.7), 47.2, 21.9,
                            boxstyle="round,pad=0,rounding_size=1.8",
                            facecolor=RED_F, edgecolor="none"))
K = ("black", "normal")
rich_text(110.0, 20.6,
          [[("There is a non-zero overlap.", *K)],
           [(" ", *K)],
           [(r"$\Rightarrow$ Input pixels", *K), ("\u00a0partially", RED, "bold")],
           [("influence", RED, "bold"), ("\u00a0each other in", *K)],
           [("the hidden representation.", *K)]],
          align="left", box_alignment=(0.0, 0.5), sep_line=5.0)

# ---- Case B: orthogonal ----------------------------------------------
ax.text(188.6, 82.5, "Case B: Orthogonal", fontsize=FS_CASE,
        fontweight="bold", color=BLUE, ha="center", va="center")
ax.text(188.6, 77.0, "(zero overlap)", fontsize=FS_CASE, fontweight="bold",
        color=BLUE, ha="center", va="center")

OBx, OBy = 173.7, 44.2
axes_triad(OBx, OBy, up=29.8, right=36.8, diag=(7.7, 8.2))
arrow(OBx, OBy, OBx, OBy + 23.8, color=RED, lw=2.6, scale=22)
ax.text(OBx + 3.4, OBy + 21.0, r"$\boldsymbol{w_p}$", fontsize=FS_VEC,
        color=RED, ha="center", va="center")
arrow(OBx, OBy, OBx + 27.8, OBy, color=GREEN, lw=2.6, scale=22)
ax.text(OBx + 24.5, OBy + 3.4, r"$\boldsymbol{w_q}$", fontsize=FS_VEC,
        color=GREEN, ha="center", va="center")
arrow(OBx, OBy, OBx - 7.0, OBy - 7.4, color=BLUE, lw=2.6, scale=22)
ax.text(OBx - 3.4, OBy - 9.4, r"$\boldsymbol{w_r}$", fontsize=FS_VEC,
        color=BLUE, ha="center", va="center")
# right-angle marker
s = 3.0
ax.plot([OBx, OBx + s, OBx + s], [OBy + s, OBy + s, OBy], color="black", lw=1.6)

ax.add_patch(FancyBboxPatch((163.5, 9.7), 48.5, 21.9,
                            boxstyle="round,pad=0,rounding_size=1.8",
                            facecolor=BLUE_F, edgecolor="none"))
rich_text(166.0, 20.6,
          [[("All columns are orthogonal.", *K)],
           [(" ", *K)],
           [(r"$\Rightarrow$ Input pixels influence", *K)],
           [("different hidden directions", *K)],
           [("independently.", BLUE, "bold")]],
          align="left", box_alignment=(0.0, 0.5), sep_line=5.0)

# ----------------------------------------------------------------------
# 4) From the whole input to the hidden representation
# ----------------------------------------------------------------------
arrow(216.8, 52.4, 225.7, 52.4, lw=3.0, scale=24)

ax.text(255.4, 76.4, "From the whole input", fontsize=FS_HEAD,
        fontweight="bold", ha="center", va="center")
ax.text(255.4, 71.0, "to hidden representation", fontsize=FS_HEAD,
        fontweight="bold", ha="center", va="center")

ax.add_patch(FancyBboxPatch((227.0, 37.8), 56.9, 25.9,
                            boxstyle="round,pad=0,rounding_size=2.0",
                            facecolor=YEL_FC, edgecolor=YEL_EC, linewidth=1.8))
ax.text(255.4, 50.7,
        r"$h(\boldsymbol{x}) = \sum_{i=1}^{d} \boldsymbol{w}_i x_i$",
        fontsize=FS_FORM, ha="center", va="center")

fig.savefig(os.path.join(OUT_DIR, "Overlap_hidden_representations.png"),
            dpi=DPI * SCALE, facecolor="white")
print("saved")
