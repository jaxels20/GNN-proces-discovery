# constants.py
import matplotlib.pyplot as plt

# Figure settings for a two-column layout (assuming ~3.5 inches per column)
DOUBLE_COL_FIG_WIDTH = 7  # total width for two columns in inches
DOUBLE_COL_FIG_HEIGHT = 4  # adjust height as needed for visibility
SINGLE_COL_FIG_WIDTH = 3.5  # width for a single column in inches
SINGLE_COL_FIG_HEIGHT = 4  # adjust height as needed for visibility
DPI = 300  # high DPI for publication quality

# Theme and Style
plt.style.use('ggplot')  # use a consistent style, or 'ggplot'/'bmh'

# Colors, Fonts, etc.
FONT_SIZE = 10
FONT_FAMILY = 'Arial'
LINE_WIDTH = 1.5
NODE_COLOR = '#1f77b4'  # example color
EDGE_COLOR = '#333333'


# Any other global settings for your figures
SHOW_GRID = False  # Often clearer without grid lines in complex graphs

OUTPUT_DIR = './figure_generation/figures/'  # Output directory for saving figures