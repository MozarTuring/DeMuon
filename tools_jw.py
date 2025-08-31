import random
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.lines import Line2D
import matplotlib as mpl


mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['lines.linestyle'] = '-'  

all_markers = list(Line2D.markers.keys())
# Sample e.g. 8 markers randomly (excluding None and ' ')
valid_markers = [m for m in all_markers if m not in [None, " ", ""]]
sampled = random.sample(valid_markers, 11)
color_sampled=plt.cm.tab20.colors[:11]
plt.rcParams['axes.prop_cycle'] = cycler(color=color_sampled, marker=sampled)
