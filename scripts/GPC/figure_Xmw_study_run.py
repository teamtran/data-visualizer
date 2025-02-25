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
if not result_path.exists():
    result_path.mkdir(parents=True)

# Data directory path
data_dir: Path = root_path / "data" / "GPC" / experiment_name

# Filenames
gpc_filenames: list[str] = [
    # "2024_06_23_PS2439_6_1K.txt",
    # TODO: need to re-run all of these on same day.
    "2025_02_11_PS_Tosoh_A1000.txt",
    "2025_02_11_PS_Tosoh_F1.txt",
    "2025_02_11_PS_Tosoh_F2.txt",
    "2025_02_11_PS_Tosoh_F4.txt",
    "2025_02_11_PS_Tosoh_F10.txt",
    "2025_02_11_005d-M1.txt",
    "2025_02_11_005d-L1.txt",
    "2025_02_11_005d-L2.txt",
    "2025_02_11_005d-L3.txt",
    "2025_02_11_005d-L4.txt",
    #     "2025_02_13_005d-M1-crude.txt",
    #     "2025_02_13_005d-L1-crude.txt",
    #     "2025_02_13_005d-L2-crude.txt",
    #     "2025_02_13_005d-L3-crude.txt",
    #     "2025_02_13_005d-L4-crude.txt",
]

labels: list = [
    "PS-1.12K",
    "PS-10K",
    "PS-19.6K",
    "PS-40.4K",
    "PS-110K",
    "PS-1.1K-SCF3",
    "PS-10K-SCF3",
    "PS-19.6K-SCF3",
    "PS-40.4K-SCF3",
    "PS-110K-SCF3",
    # "PS-1.1K-SCF3-crude",
    # "PS-10K-SCF3-crude",
    # "PS-19.6K-SCF3-crude",
    # "PS-40.4K-SCF3-crude",
    # "PS-110K-SCF3-crude",
]

gpc_metadata: list = [
    # "Mn=54.5k, Mw=121.3k, Mp=100k",  # PS-Styrofoam
    # "Mn=48.1k, Mw=140k, Mp=92.2k" # PS-Styrofoam-SCF3
    "Mn=0.7k, Mw=0.9k, Mp=1.0k",  # PS-1.12K
    "Mn=8.5k, Mw=9.2k, Mp=9.8k",  # PS-10K
    "Mn=16.7k, Mw=17.9k, Mp=19.5k",  # PS-19.6K
    "Mn=34.6k, Mw=37.4k, Mp=41.4k",  # PS-40.4K
    "Mn=94.0k, Mw=103.0k, Mp=111.2k",  # PS-110K
    "Mn=0.9k, Mw=1.1k, Mp=1.0k",  # PS-1.1K-SCF3
    "Mn=9.5k, Mw=10.2k, Mp=10.8k",  # PS-10K-SCF3
    "Mn=17.2k, Mw=18.4k, Mp=19.6k",  # PS-19.6K-SCF3
    "Mn=33.8k, Mw=36.8k, Mp=40.7k",  # PS-40.4K-SCF3
    "Mn=84.9k, Mw=100k, Mp=109k",  # PS-110K-SCF3
    # "Mn=0.9k, Mw=1.1k, Mp=1.0k",  # PS-1.1K-SCF3
    # "Mn=8.5k, Mw=9.2k, Mp=9.8k",  # PS-10K-SCF3
    # "Mn=16.6k, Mw=17.9k, Mp=19.4k",  # PS-19.6K-SCF3
    # "Mn=34.0k, Mw=36.7k, Mp=40.4k",  # PS-40.4K-SCF3
    # "Mn=91.8k, Mw=98.4k, Mp=104.7k",  # PS-110K-SCF3
]

colors: list = [
    "#d9d9d9",
    "#bdbdbd",
    "#969696",
    "#636363",
    "#252525",
    "#a6bddb",
    "#74a9cf",
    "#3690c0",
    "#0570b0",
    "#045a8d",
]

xlims: list = [(7.7, 9.7), (6.7, 8.3), (6.4, 8.0), (5.9, 7.6), (5.0, 7.5)]

inset_xlims: list = [(8.7, 9.1), (7.5, 7.7), (7.2, 7.35), (6.8, 6.9), (6.2, 6.45)]

if __name__ == "__main__":
    for i in range(len(gpc_filenames) - 5):
        gpc_plots = GPCPlots(
            data_dir=data_dir,
            gpc_data_path=[gpc_filenames[i], gpc_filenames[i + 5]],
            labels=[labels[i], labels[i + 5]],
            colors=[colors[i], colors[i + 5]],
            result_dir=result_path,
            style_path=style_path,
        )
        gpc_plots.plot_gpc(
            gpc_metadata=[gpc_metadata[i], gpc_metadata[i + 5]],
            xlim=xlims[i],
            ylim=(-0.1, 1.1),
            inset_xlim=None,
            rt=7.25,
        )
