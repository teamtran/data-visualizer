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
experiment_name: str = "AL_1_45"

# Result path
result_path: Path = root_path / "results" / "photophysical" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "photophysical" / experiment_name

# Filenames
uv_vis_filename: str = "AL_1_45_D_E_F.xlsx"
photoluminescence_filename: str = "AL_1_45_D_PL.xlsx"

uv_vis_experiment_names: list[str] = [
    "AL_1_45D_0.05mgml_water",
    "AL_1_45E_0.1mgml_water",
    "AL_1_45F_0.1mgml_water",
]
photoluminescence_experiment_names: list[str] = [
    "Emission AL-1-35L low",
    "Emission AL-1-35L med",
    "Emission AL-1-35L high",
]

labels: list = ["AL_1_45_D", "AL_1_45_E", "AL_1_45_F"]
colors = ["blue", "green", "yellow"]

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
        xlim=(200, 800),
        ylim=(-0.1, 1.1),
    )
    uv_vis_plots.plot_pl(
        drop_columns=[0, 1, 2, 3],
        normalize=True,
        baseline=True,
        xlim=(550, 800),
        ylim=(-0.1, 1.1),
    )
