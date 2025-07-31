from pathlib import Path
import sys
from matplotlib import pyplot as plt
import json

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))

from code.GPC.gpc_plots import GPCPlots

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
experiment_name: str = "005d-L_M"

# Result path
result_path: Path = root_path / "results" / "GPC" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "GPC" / experiment_name

# Filenames
gpc_filenames: list[str] = ["2025_02_11_PS_Tosoh_A1000.txt", "2025_02_11_005d-M1.txt"]

labels: list = ["PS-1.12K", "PS-1.12K-SCF$_{3}$"]

gpc_metadata: list = []

if __name__ == "__main__":
    gpc_plots = GPCPlots(
        data_dir=data_dir,
        gpc_data_path=gpc_filenames,
        labels=labels,
        colors=["#000000", "#ffcc02"],
        result_dir=result_path,
        style_path=style_path,
    )
    gpc_plots.plot_gpc(
        gpc_metadata=gpc_metadata,
        xlim=(7.75, 9.75),
        ylim=(-0.1, 1.1),
        inset_xlim=None,
        rt=7.25,
    )
