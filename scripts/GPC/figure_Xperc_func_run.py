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
experiment_name: str = "005f-I"

# Result path
result_path: Path = root_path / "results" / "GPC" / experiment_name
if not result_path.exists():
    result_path.mkdir(parents=True)

# Data directory path
data_dir: Path = root_path / "data" / "GPC" / experiment_name

# Filenames
gpc_filenames: list[str] = [
    # "2024_06_23_PS2439_6_1K.txt",
    "2025_02_11_PS_Tosoh_F10.txt",
    "2025_02_11_005d-L4.txt",
    "2025_02_13_005f-I1-flash.txt",
    "2025_02_13_005f-I2-flash.txt",
    "2025_02_13_005f-I3-flash.txt",
    "2025_02_13_005f-I4-flash.txt",
]

labels: list = [
    "PS-110K",
    "PS-110K-SCF3-0.2eq",
    "PS-110K-SCF3-0.5eq",
    "PS-110K-SCF3-1.0eq",
    "PS-110K-SCF3-2.0eq",
    "PS-110K-SCF3-2.0eq-4CzIPN-0.1eq",
    # "PS-110K-SCF3-crude",
]

gpc_metadata: list = [
    "Mn=94.0k, Mw=103.0k, Mp=111.2k",  # PS-110K
    "Mn=84.9k, Mw=100k, Mp=109k",  # PS-110K-SCF3-0.2eq
    "Mn=97.3k, Mw=104k, Mp=111k",  # PS-110K-SCF3-0.5eq
    "Mn=97.7k, Mw=105k, Mp=110k",  # PS-110K-SCF3-1.0eq
    "Mn=103k, Mw=109k, Mp=112k",  # PS-110K-SCF3-2.0eq
    "Mn=100k, Mw=106k, Mp=110k",  # PS-110K-SCF3-2.0eq-4CzIPN-0.1eq
]

colors: list = [
    "#969696",
    "#a6bddb",
    "#74a9cf",
    "#3690c0",
    "#0570b0",
    "#023858",
]


if __name__ == "__main__":
    gpc_plots = GPCPlots(
        data_dir=data_dir,
        gpc_data_path=gpc_filenames,
        labels=labels,
        colors=colors,
        result_dir=result_path,
        style_path=style_path,
    )
    gpc_plots.plot_gpc(
        gpc_metadata=gpc_metadata,
        xlim=(5.2, 7.2),
        ylim=(-0.1, 1.1),
        inset_xlim=(6.26, 6.34),
        rt=7.25,
    )
