from pathlib import Path
import sys
from matplotlib import pyplot as plt
import json

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))

from code.TGA.tga_plots import TGAPlots

# Style path
style_path: Path = root_path / "style" / "style.json"
# read style file
style: dict = json.loads(style_path.read_text())
# defines font, fontsize, and axes linewidth of all of the following plots
plt.rcParams["font.family"] = style["fontfamily"]
plt.rcParams["font.size"] = style["fontsize"]
plt.rcParams["axes.linewidth"] = style["axes_linewidth"]

# TODO: Change experiment name and filenames
# TGA Curves
# Experiment Name
experiment_name: str = "summary"

# Result path
result_path: Path = root_path / "results" / "TGA" / experiment_name
if not result_path.exists():
    result_path.mkdir(parents=True)

# Data directory path
data_dir: Path = root_path / "data" / "TGA" / experiment_name

# Filenames

if __name__ == "__main__":
    tga_summary_plots = TGAPlots(
        data_dir=data_dir,
        tga_data_path=None,
        ms_data_path=None,
        labels=[],
        colors=[],
        result_dir=result_path,
        style_path=style_path,
    )
    tga_summary_plots.plot_onset_temperature_vs_mn("mn_summary.csv")
    tga_summary_plots.plot_mass_loss_vs_mn("mn_summary.csv")

    tga_summary_plots.plot_onset_temperature_vs_percent_functionalization(
        "percent_func_summary.csv"
    )
    tga_summary_plots.plot_mass_loss_vs_percent_functionalization(
        "percent_func_summary.csv"
    )
