BASELINE_NAME = "Base"
AC_NAME = "+AC"

CMAP = {
    BASELINE_NAME: "tab:red",
    AC_NAME: "tab:blue",
}
# For single column plots (e.g. teaser)
FONTSIZE_TITLE_SC = 20
FONTSIZE_LABELS_SC = 18
FONTSIZE_TICKS_SC = 16
FONTSIZE_LEGEND_SC = 14
LINEWIDTH_SC = 1.5
MARKERSIZE_SC = 100

# For double column plots
FONTSIZE_TITLE = 22
FONTSIZE_LABELS = 20
FONTSIZE_TICKS = 18
FONTSIZE_LEGEND = 15
LINEWIDTH = 1.5
MARKERSIZE = 20
ALPHA = 0.1

METHODS = {
    "ancl_tw_ex_0": "ANCL",
    # "ancl_tw_ex_2000": "ANCL+Ex,
    "bic": "BiC",
    "ewc": "EWC",
    "er": "ER",
    "der++": "DER++",
    "finetuning_ex0": "FT",
    "finetuning_ex2000": "FT+Ex",
    "gdumb": "GDumb",
    "lode": "LODE",
    # "icarl": "iCaRL",
    "lwf": "LwF",
    # "joint": "Joint",
    "ssil": "SSIL",
}

METHOD_TO_COLOR = {
    "ANCL": "red",  # red
    "BiC": "purple",  # yellow
    "DER++": "orange",  # purple
    "LODE": "blue",  # green
    "SSIL": "green",  # blue

    "FT": "brown",  # purple
    "FT+Ex": "pink",  # red
    "ER": "darkblue",  # orange
    "EWC": "olive",  # green
    "GDumb": "darkred",  # olive
    "LwF": "lightgreen",  # blue
    "Joint": 'gray', # gray
}

FIGSIZE_MAIN_PAPER= (12, 4)
FIGSIZE_APPENDIX_JOINT_PLOTS= (16, 4)
FIGSIZE_ABLATIONS = (12, 3)