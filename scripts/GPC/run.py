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
experiment_name: str = "005c"

# Result path
result_path: Path = root_path / "results" / "GPC" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "GPC" / experiment_name

# Filenames
gpc_filenames: list[str] = [
    "2024_06_23_PS2439_6_1K.txt",
    "2024_06_23_005c-A-3.txt",
    "2024_06_23_005c-B-2.txt",
    "2024_06_23_005c-C.txt",
    "2024_06_23_005c-E.txt",
]

labels: list = [
    "PS-6.1K",
    "Water + O2",
    "No Water + No O2",
    "Water + No O2",
    "No Water + O2",
]

gpc_metadata: list = [
    "Mn=5127, Mw=6706, D=1.308",
    "Mn=5895, Mw=6927, D=1.175",
    "Mn=6109, Mw=7537, D=1.234",
    "Mn=5752, Mw=7277, D=1.265",
    "Mn=5943, Mw=7565, D=1.273",
]

if __name__ == "__main__":
    gpc_plots = GPCPlots(
        data_dir=data_dir,
        gpc_data_path=gpc_filenames,
        labels=labels,
        colors=["#8286ff", "#00ff00", "#f5c92a", "#AF69EE", "#FFC0CB"],
        result_dir=result_path,
        style_path=style_path,
    )
    gpc_plots.plot_gpc(gpc_metadata=gpc_metadata, xlim=(6.5, 9.5), ylim=(-0.1, 1.1))
