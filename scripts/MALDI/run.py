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
experiment_name: str = "005d-M1"

# Result path
result_path: Path = root_path / "results" / "MALDI" / experiment_name
if not result_path.exists():
    result_path.mkdir(parents=True)

# Data directory path
data_dir: Path = root_path / "data" / "MALDI" / experiment_name

# Filenames
maldi_filenames: list[str] = [
    # "2025_01_23_PS_Tosoh_A1000_dilute_700_3000_0_M4_1.txt",
    # "2024_10_17_005f-A1_0_E11_1.txt",
    "2025_01_23_005d-M1_dilute_700_3000_0_M5_1.txt",
    # "2024_10_17_005f-A3_0_E13_1.txt",
    # "2024_10_17_PS_Tosoh_F2_0_E6_1.txt",
    # "2024_10_17_PS_Tosoh_F4_0_E7_1.txt",
]

labels: list = [
    # "PS-1K",
    # "005f-A1",
    "PS-1K-SCF$_3$",
    # "005f-A3",
    # "PS-F2-20K",
    # "PS-F4-40K"
]

if __name__ == "__main__":
    maldi_plots = MALDIPlots(
        data_dir=data_dir,
        ms_data_path=maldi_filenames,
        labels=labels,
        colors=["#ad8c09", "#e0f3db", "#7bccc4"],  # "#8286ff"
        result_dir=result_path,
        style_path=style_path,
        opacity=[1, 1],
    )
    maldi_plots.plot_maldi(xlim=(800, 3000))
    # Trial and error with prominence to get the peaks that you want detected
    maldi_plots.plot_maldi_zoom(xlim=(880, 1010), ylim=(0, 1), prominence=10)
    # maldi_plots.plot_maldi_zoom_with_inset(
    #     xlim=(850, 1020), inset_xlim=(990, 1010), prominence=10
    # )
