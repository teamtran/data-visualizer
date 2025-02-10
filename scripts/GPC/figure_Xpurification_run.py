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
experiment_name: str = "005d-B"

# Result path
result_path: Path = root_path / "results" / "GPC" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "GPC" / experiment_name

# Filenames
gpc_filenames: list[str] = [
    # "2024_06_23_PS2439_6_1K.txt",
    # TODO: need to re-run all of these on same day.
    "2025_01_17_PS_6_1K.txt",
    "2025_01_17_005d-B1-FLASH.txt",
    "2025_01_17_005d-B1-rGPC.txt",
    "2025_01_17_005d-B1-toyopearls.txt",
]

labels: list = [
    "PS-6.4K",
    "PS-SCF3-FLASH",
    "PS-SCF3-rGPC",
    "PS-SCF3-Toyopearls",
]

gpc_metadata: list = [
    "Mn=5010, Mw=6352, Mp=8118",
    "Mn=6335, Mw=7613, Mz=8291",
    "Mn=6977, Mw=8975, Mp=8493",
    "Mn=6395, Mw=8320, Mp=8392",
]

if __name__ == "__main__":
    gpc_plots = GPCPlots(
        data_dir=data_dir,
        gpc_data_path=gpc_filenames,
        labels=labels,
        colors=["#8286ff", "#f5c92a", "#e0f3db", "#7bccc4"],
        result_dir=result_path,
        style_path=style_path,
    )
    gpc_plots.plot_gpc(
        gpc_metadata=gpc_metadata,
        xlim=(6.75, 8.5),
        ylim=(-0.1, 1.1),
        inset_xlim=(7.65, 7.82),
        rt=7.25,
    )
