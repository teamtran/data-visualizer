from pathlib import Path
import sys
from matplotlib import pyplot as plt
import json

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))

from code.MALDI.maldi_plots import MALDIPlots

# Style path
style_path: Path = root_path / "style" / "style.json"
# read style file
style: dict = json.loads(style_path.read_text())
# defines font, fontsize, and axes linewidth of all of the following plots
plt.rcParams["font.family"] = style["fontfamily"]
plt.rcParams["font.size"] = style["fontsize"]
plt.rcParams["axes.linewidth"] = style["axes_linewidth"]

# TODO: Change experiment name and filenames
# GPC Curves
# Experiment Name
experiment_name: str = "PS_Tosoh_Standards"

# Result path
result_path: Path = root_path / "results" / "MALDI" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "MALDI" / experiment_name

# Filenames
maldi_filenames: list[str] = [
    "2024_10_03_PS_Tosoh_F4_sum_0_G19_1.txt",
]

labels: list = ["PS-F4-40K"]

if __name__ == "__main__":
    maldi_plots = MALDIPlots(
        data_dir=data_dir,
        ms_data_path=maldi_filenames,
        labels=labels,
        colors=["#8286ff", "#00ff00", "#f5c92a", "#AF69EE", "#FFC0CB"],
        result_dir=result_path,
        style_path=style_path,
    )
    maldi_plots.plot_maldi(xlim=(25000, 50000))
    # Trial and error with prominence to get the peaks that you want detected
    # maldi_plots.plot_maldi_zoom(xlim=(9500, 10500), prominence=0.05)
