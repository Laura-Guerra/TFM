import matplotlib as mpl
import altair as alt

COLORS = {
    "dark_blue": "#00004E",
    "blue": "#0043AF",
    "light_blue": "#73BAFF",
    "grey": "#DFE6EA",
    "red": "#FF3333",
    "yellow": "#FFCD1B",
    "black": "#000000",
    "white": "#FFFFFF",
}


def set_matplotlib_theme():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.facecolor": COLORS["white"],
        "figure.facecolor": COLORS["white"],
        "axes.edgecolor": COLORS["grey"],
        "axes.labelcolor": COLORS["black"],
        "xtick.color": COLORS["black"],
        "ytick.color": COLORS["black"],
        "text.color": COLORS["black"],
        "axes.grid": True,
        "grid.color": COLORS["grey"],
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.loc": "best",
        "figure.figsize": (10, 6),
        "lines.linewidth": 1,
        "lines.markersize": 6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.prop_cycle": mpl.cycler(color=[
            COLORS["dark_blue"],
            COLORS["blue"],
            COLORS["light_blue"],
            COLORS["red"],
            COLORS["yellow"]
        ])

    })

def altair_custom_theme():
    def theme():
        return {
            "config": {
                "background": COLORS["white"],
                "title": {
                    "fontSize": 14,
                    "font": "DejaVu Sans",
                    "color": COLORS["black"]
                },
                "axis": {
                    "labelColor": COLORS["black"],
                    "titleColor": COLORS["black"],
                    "gridColor": COLORS["grey"],
                    "gridOpacity": 0.5,
                    "tickColor": COLORS["black"]
                },
                "legend": {
                    "labelColor": COLORS["black"],
                    "titleColor": COLORS["black"]
                },
                "view": {
                    "stroke": COLORS["grey"]
                },
                "range": {
                    "category": [
                        COLORS["dark_blue"],
                        COLORS["blue"],
                        COLORS["light_blue"],
                        COLORS["red"],
                        COLORS["yellow"],
                        COLORS["orange"],
                        COLORS["green"],
                        COLORS["purple"],
                        COLORS["pink"],
                        COLORS["teal"]
                    ]
                }
            }
        }
    return theme

def set_custom_theme():
    set_matplotlib_theme()
    alt.themes.register("tfm_theme", altair_custom_theme())
    alt.themes.enable("tfm_theme")
