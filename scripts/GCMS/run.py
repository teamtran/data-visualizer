from pathlib import Path
import sys
from matplotlib import pyplot as plt
import json

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))

from code.GCMS.gcms_plots import GCMS_LinearCalibration_Plots, MSPlots, GCMSPlots

# Style path
style_path: Path = root_path / "style" / "style.json"
# read style file
style: dict = json.loads(style_path.read_text())
# defines font, fontsize, and axes linewidth of all of the following plots
plt.rcParams["font.family"] = style["fontfamily"]
plt.rcParams["font.size"] = style["fontsize"]
plt.rcParams["axes.linewidth"] = style["axes_linewidth"]

# TODO: Change experiment name and filenames
# GCMS Curves
# Experiment Name
experiment_name: str = "005c-H"

# Result path
result_path: Path = root_path / "results" / "GCMS" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "GCMS" / experiment_name

# Filenames
gcms_lin_cal_filename: str = "isopropylbenzene-SCF3-linear-calibration.csv"
# gcms_filename: str = [
#     "2024_12_16_005c-H1F_TIC.txt",
#     "2024_12_16_005c-H1F_XIC_266.txt",
#     "2024_12_16_005c-H1F_XIC_364.txt",
#     "2024_12_16_005c-H1F_XIC_366.txt",
# ]
gcms_filename: list[str] = [
    "2024_12_16_005c-H3F_TIC.txt",
    "2024_12_16_005c-H3F_XIC_266.txt",
    "2024_12_16_005c-H3F_XIC_364.txt",
    "2024_12_16_005c-H3F_XIC_366.txt",
]
ms_filename: str = "2024_12_16_005c-H1F_t_3_8m.txt"
# label = "Isopropylbenzene-SCF3"
# labels = ["005c-H1F", "005c-H1F_XIC_266", "005c-H1F_XIC_364", "005c-H1F_XIC_366 "]
labels = ["005c-H3F", "005c-H3F_XIC_266", "005c-H3F_XIC_364", "005c-H3F_XIC_366 "]
if __name__ == "__main__":
    # gcms_linear_cal_plot = GCMS_LinearCalibration_Plots(
    #     data_dir=data_dir,
    #     gcms_filename=gcms_filename,
    #     label=label,
    #     color="#8286ff",
    #     result_dir=result_path,
    #     style_path=style_path,
    #     nm=230,
    # )
    # gcms_linear_cal_plot.plot_calibration_curve()
    # gcms_plots = GCMSPlots(
    #     data_dir=data_dir,
    #     gcms_data_path=gcms_filename,
    #     result_dir=result_path,
    #     style_path=style_path,
    #     gcms_type="TIC",
    #     labels=labels,
    #     colors=["#8286ff", "#00ff00", "#f5c92a", "#AF69EE", "#FFC0CB"],
    # )
    # gcms_plots.plot_gcms(xlim=(2.5, 7))
    ms_plots = MSPlots(
        data_dir=data_dir,
        ms_data_path=ms_filename,
        result_dir=result_path,
        style_path=style_path,
    )
    ms_plots.plot_ms(time=3.8, xlim=(150, 400))
