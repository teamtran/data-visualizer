from pathlib import Path
import sys
from matplotlib import pyplot as plt
import json

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))

from code.LCMS.lcms_plots import LCMS_LinearCalibration_Plots

# Style path
style_path: Path = root_path / "style" / "style.json"
# read style file
style: dict = json.loads(style_path.read_text())
# defines font, fontsize, and axes linewidth of all of the following plots
plt.rcParams["font.family"] = style["fontfamily"]
plt.rcParams["font.size"] = style["fontsize"]
plt.rcParams["axes.linewidth"] = style["axes_linewidth"]

# TODO: Change experiment name and filenames
# LCMS Curves
# Experiment Name
experiment_name: str = "005b-D"

# Result path
result_path: Path = root_path / "results" / "LCMS" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "LCMS" / experiment_name

# Filenames
lcms_filename: str = "isopropylbenzene-SCF3-linear-calibration.csv"

label = "Isopropylbenzene-SCF$_3$-Bpin"

if __name__ == "__main__":
    lcms_linear_cal_plot = LCMS_LinearCalibration_Plots(
        data_dir=data_dir,
        lcms_filename=lcms_filename,
        label=label,
        color="#8286ff",
        result_dir=result_path,
        style_path=style_path,
        nm=230,
    )
    lcms_linear_cal_plot.plot_calibration_curve()
