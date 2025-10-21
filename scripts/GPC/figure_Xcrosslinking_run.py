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
experiment_name: str = "005h"

# Result path
result_path: Path = root_path / "results" / "GPC" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "GPC" / experiment_name

# Filenames
gpc_filenames: list[str] = [
    # "2024_10_29_PS_Tosoh_F1.txt",
    "2024_11_01_PS_Tosoh_F1.txt",
    # "2024_10_14_PS_Tosoh_F1.txt",
    # "2024_10_16_005f-A1-crude.txt",
    # "2024_10_16_005f-A2-crude.txt",
    # "2024_10_16_005f-A3-crude.txt",
    "2024_10_28_005h-A3-crude.txt",
    # "2024_10_28_005h-A1-crude.txt",
    # "2024_10_28_005h-A2-crude.txt",
    # "2024_10_28_005h-A4-crude.txt",
    "2024_10_29_005h-B1-crude.txt",
    # "2024_10_29_005h-B4-crude.txt",
    # "2024_10_29_005h-B2-crude.txt",
    # "2024_10_30_005h-C1-crude.txt",
    # "2024_10_30_005h-C2-crude.txt",
    # "2024_10_31_005h-D2-crude.txt",
    # "2024_10_31_005h-D1-crude.txt",
    # "2024_11_01_005h-B4-crude.txt",
    "2024_11_01_005h-B2-crude.txt",
    "2024_10_29_005h-B3-crude.txt",
    "2024_11_01_005h-C1-crude.txt",
    "2024_11_01_005h-C2-crude.txt",
    "2024_11_01_005h-D2-crude.txt",
    # "2024_11_01_005h-D1-crude.txt",
]

labels: list = [
    "PS-10K",
    # "No deviation",
    # "0.125eq_Phth-SCF3",
    # "0.0625eq_Phth-SCF3",
    # "0.25eq_Phth-SCF3",
    # "16hr_75_DCE",
    # "16hr_75_DCE_2x_vol",
    # "16hr_no_phth-scf3",
    # "16hr_0.0125eq Base",
    # "2hr",
    # "4hr_0.125eq Base",
    # "4hr_0.25eq Base",
    # "4hr_0.025eq PC (10x)",
    # "4hr_0.5eq Base",
    # "4hr_1eq Base",
    # "4hr_6eq Base",
    # "4hr_12eq Base",
    "Entry 1 (control)",
    "Entry 2",
    "Entry 3",
    "Entry 4",
    "Entry 5",
    "Entry 6",
    "Entry 7",
]

gpc_metadata: list = []

if __name__ == "__main__":
    gpc_plots = GPCPlots(
        data_dir=data_dir,
        gpc_data_path=gpc_filenames,
        labels=labels,
        colors=[
            "#8286ff",
            "#2166ac",
            "#a50026",
            "#d73027",
            "#f46d43",
            "#fdae61",
            "#fee08b",
            "#d9ef8b",
            "#a6d96a",
            "#66bd63",
            "#1a9850",
            "#006837",
        ],
        result_dir=result_path,
        style_path=style_path,
    )
    gpc_plots.plot_gpc(
        gpc_metadata=gpc_metadata,
        xlim=(6.5, 8.5),
        ylim=(-0.1, 1.1),
        inset_xlim=(7.58, 7.68),
        rt=7.25,
    )
