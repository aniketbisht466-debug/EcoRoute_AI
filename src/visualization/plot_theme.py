
"""
EcoRoute AI
Scientific Visualization Theme
Version 1.0
"""

import matplotlib.pyplot as plt

COLORS = {
    "background": "#0B1220",
    "surface": "#111C2E",
    "eco": "#10B981",
    "baseline": "#38BDF8",
    "fuel": "#F59E0B",
    "warning": "#F43F5E",
    "purple": "#A78BFA",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
    "grid": "#263449"
}


def apply_theme():

    plt.rcParams.update({

        "figure.facecolor": COLORS["background"],
        "axes.facecolor": COLORS["surface"],
        "savefig.facecolor": COLORS["background"],

        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "axes.edgecolor": COLORS["grid"],

        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],

        "grid.color": COLORS["grid"],
        "grid.alpha": 0.45,
        "grid.linestyle": "--",

        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "font.family": "DejaVu Sans",
        "font.size": 11,

        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,

        "legend.facecolor": COLORS["surface"],
        "legend.edgecolor": COLORS["grid"],

        "figure.dpi": 120,
        "savefig.dpi": 300,

        "figure.autolayout": True
    })


def save_plot(fig, path):

    fig.text(
        0.99,
        0.015,
        "ECOROUTE AI | Scientific Analytics",
        ha="right",
        fontsize=8,
        color=COLORS["muted"]
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
