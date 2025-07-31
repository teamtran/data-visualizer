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
experiment_name: str = "005c"

# Result path
result_path: Path = root_path / "results" / "TGA" / experiment_name
if not result_path.exists():
    result_path.mkdir(parents=True)

# Data directory path
data_dir: Path = root_path / "data" / "TGA" / experiment_name

# Filenames
isothermal_filenames: list = [
    "ExpDat_2024_06_21_PS2439_6_1K_300C_isotherm.csv",
    "ExpDat_2024_06_21_005c-B_300C_isotherm.csv",
]
isothermal_ms_filenames: list = [
    "ExpDat_2024_06_21_PS2439_6_1K_300C_isotherm-aeolos.csv",
    "ExpDat_2024_06_21_005c-B_300C_isotherm-aeolos.csv",
]

labels: list = ["PS", "PS-SCF3"]
# labels: list = ["PS-SCF3-1K_min", "PS-6.1K-1K_min"]
# labels: list = ["PS-SCF3", "PS-6.1K"]
# labels: list = ["PS-SCF3-0_1K_min", "PS-6_1K-0_1K_min"]
# labels: list = ["PS-1.12K", "PS-1.12K-SCF3"]
# labels: list = ["PS-10K", "PS-10K-SCF3"]
# labels: list = ["PS-19.6K", "PS-19.6K-SCF3"]
# labels: list = ["PS-40.4K", "PS-40.4K-SCF3"]
# labels: list = ["PS-110K", "PS-110K-SCF3"]

if __name__ == "__main__":
    isothermal_tga_plots = TGAPlots(
        data_dir=data_dir,
        tga_data_path=isothermal_filenames,
        ms_data_path=isothermal_ms_filenames,
        labels=labels,
        colors=["#000000", "#ffcc02"],
        result_dir=result_path,
        style_path=style_path,
    )
    isothermal_tga_plots.plot_ms_peak_area(
        "isothermal",
        300,
        initial_time=0,
        end_time=1500,
        m_z_start=60,
        m_z_end=150,
        alpha_list=[0.7, 0.9],
    )
