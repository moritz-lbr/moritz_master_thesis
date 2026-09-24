#!/usr/bin/env python3
"""
Generate the two one-hidden-layer MLP figures.

    mlp_plain.py  ->  MLP_z.png                   all connections equal
                      MLP_z_neuron_connections.png  weights of one neuron picked out

Both are written at exactly 1448 x 1086 px (4:3), the same format as the
existing MLP.png / MLP_neuron_connections.png, so dropping them into the
LaTeX document does not change any layout.

    \\includegraphics[width=\\textwidth]{MLP_z.png}

Requires: matplotlib, numpy.
"""
import os

import matplotlib
OUT_DIR = os.path.dirname(os.path.abspath(__file__))  # figures are saved next to this script
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ----------------------------------------------------------------- format ---
# 7.24 in x 200 dpi = 1448 px, 5.43 in x 200 dpi = 1086 px
FIG_W_IN, FIG_H_IN, DPI = 7.24, 5.43, 200
BG = '#FDFDFD'

# maths set in Computer Modern so the labels match a standard LaTeX document
plt.rcParams.update({
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
})

# ------------------------------------------------------------------ style ---
NODE = {                                   # fill, edge
    'input':  ('#DFE4F5', '#6572B8'),
    'hidden': ('#E7F5F2', '#73BFB1'),
    'output': ('#E4DDF3', '#8564BE'),
}
EDGE_LW   = 1.6                            # node outline
WIRE      = '#686771'                      # ordinary connection
WIRE_LW   = 0.9
WIRE_FADE = '#C8C8CE'                      # connection pushed to the background
WIRE_FADE_LW = 0.8
BOLD      = '#19233D'                      # highlighted connection
BOLD_LW   = 2.8
LABEL_C   = '#1A1A1A'

# --------------------------------------------------------------- geometry ---
# data coordinates run 0..4 by 0..3 with an equal aspect, so the axes fill the
# 4:3 canvas exactly and the nodes stay perfectly circular
XLIM, YLIM = (0.0, 4.0), (0.0, 3.0)
COLX = {'input': 0.75, 'hidden': 2.00, 'output': 3.25}
ROWY = [2.28, 1.56, 0.48]                  # first, second, last visible node
DOTY = 1.00                                # vertical ellipsis
R = 0.20                                   # node radius
HEADER_Y = 2.76

COLUMNS = {
    'input':  [r'$x_1$', r'$x_2$', r'$x_d$'],
    'hidden': [r'$z_1^{\,l}$', r'$z_2^{\,l}$', r'$z_n^{\,l}$'],
    'output': [r'$f_1$', r'$f_2$', r'$f_p$'],
}
HEADERS = {'input': r'$\vec{x}$', 'hidden': r'$\vec{z}^{\,l}$', 'output': r'$\vec{f}$'}

NODE_FS, HEADER_FS = 15, 19


def positions():
    return {c: [(COLX[c], y) for y in ROWY] for c in COLX}


def draw(highlight=None, out='figure.png'):
    """highlight: index of the hidden node whose weights are drawn bold,
    or None for the plain figure."""
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(BG)

    pos = positions()
    faded = highlight is not None

    # ---- connections, drawn first so the nodes sit on top -------------------
    for a, b in (('input', 'hidden'), ('hidden', 'output')):
        for i, p in enumerate(pos[a]):
            for j, q in enumerate(pos[b]):
                bold = False
                if highlight is not None:
                    bold = (b == 'hidden' and j == highlight) or \
                           (a == 'hidden' and i == highlight)
                if bold:
                    continue                       # drawn afterwards, on top
                ax.plot([p[0], q[0]], [p[1], q[1]],
                        color=WIRE_FADE if faded else WIRE,
                        lw=WIRE_FADE_LW if faded else WIRE_LW,
                        solid_capstyle='round', zorder=1)

    if highlight is not None:
        for a, b in (('input', 'hidden'), ('hidden', 'output')):
            for i, p in enumerate(pos[a]):
                for j, q in enumerate(pos[b]):
                    if (b == 'hidden' and j == highlight) or \
                       (a == 'hidden' and i == highlight):
                        ax.plot([p[0], q[0]], [p[1], q[1]], color=BOLD,
                                lw=BOLD_LW, solid_capstyle='round', zorder=2)

    # ---- nodes --------------------------------------------------------------
    for col, labels in COLUMNS.items():
        fill, edge = NODE[col]
        for (x, y), lab in zip(pos[col], labels):
            ax.add_patch(Circle((x, y), R, facecolor=fill, edgecolor=edge,
                                lw=EDGE_LW, zorder=3))
            ax.text(x, y, lab, ha='center', va='center',
                    fontsize=NODE_FS, color=LABEL_C, zorder=4)
        # vertical ellipsis between the second and the last node
        ax.text(COLX[col], DOTY, r'$\vdots$', ha='center', va='center',
                fontsize=20, color=LABEL_C, zorder=4)
        ax.text(COLX[col], HEADER_Y, HEADERS[col], ha='center', va='center',
                fontsize=HEADER_FS, color=LABEL_C, zorder=4)

    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    # store without an alpha channel, matching the existing MLP.png exactly
    from PIL import Image
    Image.open(out).convert('RGB').save(out)
    im = Image.open(out)
    print(f'wrote {out}  {im.size[0]}x{im.size[1]} {im.mode}')


if __name__ == '__main__':
    draw(highlight=None, out=os.path.join(OUT_DIR, 'MLP_z.png'))
    draw(highlight=1, out=os.path.join(OUT_DIR, 'MLP_z_neuron_connections.png'))   # second hidden node
