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

# Data directory path
data_dir: Path = root_path / "data" / "TGA" / experiment_name

# Filenames
isothermal_filenames: list = [
    # "ExpDat_2024_06_21_005c-B_300C_isotherm.csv",
    # "ExpDat_2024_06_21_PS2439_6_1K_300C_isotherm.csv",
    # "ExpDat_2024_06_18_005c-B_350C_isotherm.csv",
    # "ExpDat_2024_06_17_PS2439_6_1K_350C_isotherm.csv",
    "ExpDat_2024_07_02_005c-B_260C_isotherm.csv",
    "ExpDat_2024_07_02_PS2439_6_1K_260C_isotherm.csv",
]
isothermal_ms_filenames: list = [
    # "ExpDat_2024_06_21_005c-B_300C_isotherm-aeolos.csv",
    # "ExpDat_2024_06_21_PS2439_6_1K_300C_isotherm-aeolos.csv",
    # "ExpDat_2024_06_18_005c-B_350C_isotherm-aeolos.csv",
    # "ExpDat_2024_06_17_PS2439_6_1K_350C_isotherm-aeolos.csv",
    "ExpDat_2024_07_02_005c-B_260C_isotherm-aeolos.csv",
    "ExpDat_2024_07_02_PS2439_6_1K_260C_isotherm-aeolos.csv",
]

dynamic_filenames: list = [
    # "ExpDat_2024_06_20_005c-B_dynamic.csv",
    # "ExpDat_2024_06_20_PS2439_6_1K_dynamic.csv",
    "ExpDat_2024_06_24_005c-B_dynamic-slow.csv",
    "ExpDat_2024_06_28_PS2439_6_1K_dynamic-0_1K_min.csv",
]
dynamic_ms_filenames: list = [
    # "ExpDat_2024_06_20_005c-B_dynamic-aeolos.csv",
    # "ExpDat_2024_06_20_PS2439_6_1K_dynamic-aeolos.csv",
    "ExpDat_2024_06_24_005c-B_dynamic-slow-aeolos.csv",
    "ExpDat_2024_06_28_PS2439_6_1K_dynamic-0_1K_min-aeolos.csv",
]

# labels: list = ["PS-SCF3", "PS-6.1K"]
# labels: list = ["PS-SCF3", "PS-6.1K", "PS-SCF3-slow", "PS-6.1K-slow"]
labels: list = ["PS-SCF3-0_1K_min", "PS-6_1K-0_1K_min"]

if __name__ == "__main__":
    isothermal_tga_plots = TGAPlots(
        data_dir=data_dir,
        tga_data_path=isothermal_filenames,
        ms_data_path=isothermal_ms_filenames,
        labels=labels,
        colors=["#f5c92a", "#8286ff"],
        result_dir=result_path,
        style_path=style_path,
    )
    # TODO: must change temp and xlimit depending on the isothermal conditions!
    isothermal_tga_plots.plot_tga_isothermal(
        isothermal_temp=260,
        xlim=(0, 3000),
        ylim=(60, 100),
        target_mass=104,
        initial_correction_time=250,
    )
    # isothermal_tga_plots.plot_ms_peak_area(
    #     "isothermal", 300, initial_time=40, end_time=3000, m_z_start=60, m_z_end=150
    # )
    # dynamic_tga_plots = TGAPlots(
    #     data_dir=data_dir,
    #     tga_data_path=dynamic_filenames,
    #     ms_data_path=dynamic_ms_filenames,
    #     labels=labels,
    #     colors=["#f5c92a", "#8286ff", "#ad8c09", "#4772ff"],
    #     result_dir=result_path,
    #     style_path=style_path,
    # )
    # dynamic_tga_plots.plot_tga_dynamic(
    #     t_depolymerization_cutoff=99.5,
    #     target_mass=104,
    #     xlim=(220, 350),
    #     initial_correction_temp=230,
    # )
