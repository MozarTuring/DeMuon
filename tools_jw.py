import random
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.lines import Line2D
import matplotlib as mpl


mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['lines.linestyle'] = '-'  

all_markers = list(Line2D.markers.keys())
# Sample e.g. 8 markers randomly (excluding None and ' ')
markers = ["o", "s", "D", "^", "v"]
valid_markers = markers + [m for m in all_markers if m not in [None, " ", ""]+markers]
