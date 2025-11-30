from pathlib import Path
import sys
from matplotlib import pyplot as plt
import json
import pandas as pd
import numpy as np

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))

from code.LCMS.lcms_plots import LCMSPlots

# Style path
style_path: Path = root_path / "style" / "style.json"
# read style file
style: dict = json.loads(style_path.read_text())
# defines font, fontsize, and axes linewidth of all of the following plots
plt.rcParams["font.family"] = style["fontfamily"]
plt.rcParams["font.size"] = style["fontsize"]
plt.rcParams["axes.linewidth"] = style["axes_linewidth"]

# Example with multiple chromatograms for stacking
experiment_name: str = "AL"

# Result path
result_path: Path = root_path / "results" / "LCMS" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "LCMS" / experiment_name

# For stacked plot - multiple files example
lcms_filenames = [
    "2025-07-30-AL-1-TIT.dx_DAD1C_005.CSV",
    # Add more files here for stacking
    "2025-07-30-AL-1-65A-bTEG-PPP.dx_DAD1C_005.CSV",
    "2025-07-21-SL-AL-1-66B-run-Bperc-10--90--10.dx_DAD1C_005.CSV",
    # "2025-07-30-AL-2-sample.dx_DAD1B.CSV",
    # "2025-07-30-AL-3-control.dx_DAD1B.CSV",
]

labels = ["TIT", "DPP", "Degraded DPP-TIT"]  # Add corresponding labels
colors = ["#8286ff", "#5445ff", "#1004ff"]  # Add corresponding colors

if __name__ == "__main__":
    # Note: Updated parameter names to match your class definition
    lcms_plot = LCMSPlots(
        data_dir=data_dir,
        lcms_data_path=lcms_filenames,  # Note: parameter name from class def
        labels=labels,  # Note: parameter name from class def
        colors=colors,  # Note: parameter name from class def
        result_dir=result_path,
        style_path=style_path,
    )

    # Use the new stacked plot function
    lcms_plot.plot_stacked_lcms(
        xlim=(0, 12),  # Adjust as neededs
        vertical_spacing=1.6,  # Adjust spacing between chromatograms
        nm=210,
        show_legend=True,
        normalize_individual=True,  # Each chromatogram normalized to [0,1]
    )

# For multiple chromatograms example:
# lcms_filenames = [
#     "sample1.CSV",
#     "sample2.CSV",
#     "sample3.CSV"
# ]
# labels = ["Sample 1", "Sample 2", "Sample 3"]
# colors = ["#8286ff", "#ff6b6b", "#4ecdc4"]
