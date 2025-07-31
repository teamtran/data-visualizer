from pathlib import Path
import sys
from matplotlib import pyplot as plt
import json

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))

from code.photophysical.photophysical_plots import PhotophysicalPlots

# Style path
style_path: Path = root_path / "style" / "style.json"
# read style file
style: dict = json.loads(style_path.read_text())
# defines font, fontsize, and axes linewidth of all of the following plots
plt.rcParams["font.family"] = style["fontfamily"]
plt.rcParams["font.size"] = style["fontsize"]
plt.rcParams["axes.linewidth"] = style["axes_linewidth"]

# TODO: Change experiment name and filenames
# uv_vis Curves
# Experiment Name
experiment_name: str = "005d-B"

# Result path
result_path: Path = root_path / "results" / "photophysical" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "photophysical" / experiment_name

# Filenames
uv_vis_filename: str = "2024_12_18_005d-B1_purification.csv"
photoluminescence_filename: str = (
    "2025_01_30_005d-B1-purification-photoluminescence.xlsx"
)

uv_vis_experiment_names: list[str] = [
    "0.05mg_mL_4CzIPN",
    "0.965mg_mL_PS_6_1K",
    "1.09mg_mL_005d-B1-FLASH",
    # "0.985mg_mL_005d-B1-rGPC",
    # "0.95mg_mL_005d-B1-toyopearls",
]
photoluminescence_experiment_names: list[str] = [
    "0.05mg_mL_4CzIPN",
    "0.965mg_mL_PS_6_1K",
    # "1.09mg_mL_005d-B1-FLASH",
    # "0.985mg_mL_005d-B1-rGPC",
    # "0.95mg_mL_005d-B1-toyopearls",
]

labels: list = [
    "4CzIPN",
    "PS",
    "PS-SCF3-FLASH",
    # "PS-SCF3-rGPC",
    # "PS-SCF3-Toyopearls",
]
colors = ["#e6e600", "#000000", "#F4BD14"]

if __name__ == "__main__":
    uv_vis_plots = PhotophysicalPlots(
        data_dir=data_dir,
        uv_vis_data_path=uv_vis_filename,
        photoluminescence_data_path=photoluminescence_filename,
        uv_vis_experiment_names=uv_vis_experiment_names,
        photoluminescence_experiment_names=photoluminescence_experiment_names,
        labels=labels,
        colors=colors,
        result_dir=result_path,
        style_path=style_path,
    )
    uv_vis_plots.plot_uv_vis_and_pl(
        drop_columns=[0, 1, 2, 3],
        normalize=True,
        baseline=True,
        xlim=(200, 800),
        ylim=(-0.1, 1.1),
    )
    uv_vis_plots.plot_uv_vis(
        drop_columns=[0, 1, 2, 3],
        normalize=True,
        baseline=True,
        xlim=(200, 600),
        ylim=(-0.1, 1.1),
    )
    uv_vis_plots.plot_pl(
        drop_columns=[0, 1, 2, 3],
        normalize=True,
        baseline=True,
        xlim=(365, 700),
        ylim=(-0.1, 1.1),
    )
