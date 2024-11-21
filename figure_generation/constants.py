# constants.py
import matplotlib.pyplot as plt

# Figure settings for a two-column layout (assuming ~3.5 inches per column)
FIG_WIDTH = 7  # total width for two columns in inches
FIG_HEIGHT = 4  # adjust height as needed for visibility

DPI = 300  # high DPI for publication quality

# Theme and Style
plt.style.use('ggplot')  # use a consistent style, or 'ggplot'/'bmh'

# Colors, Fonts, etc.
FONT_SIZE = 12
FONT_FAMILY = 'Arial'
LINE_WIDTH = 1.5


# Any other global settings for your figures
SHOW_GRID = False  # Often clearer without grid lines in complex graphs

OUTPUT_DIR = './figure_generation/figures/'  # Output directory for saving figures