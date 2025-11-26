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
experiment_name: str = "005d-L_M"

# Result path
result_path: Path = root_path / "results" / "TGA" / experiment_name
if not result_path.exists():
    result_path.mkdir(parents=True)

# Data directory path
data_dir: Path = root_path / "data" / "TGA" / experiment_name

# Filenames
isothermal_filenames: list = [
    "ExpDat_2025_07_04_SL_PS_Tosoh_F1_isothermal.csv",
    "ExpDat_2025_03_14_SL_005D_L1_isothermal.csv",
]
isothermal_ms_filenames: list = [
    "ExpDat_2025_07_04_SL_PS_Tosoh_F1_isothermal-aeolos.csv",
    "ExpDat_2025_03_14_SL_005D_L1_isothermal-aeolos.csv",
]

dynamic_filenames: list = [
    # "ExpDat_2024_06_20_005c-B_dynamic.csv",
    # "ExpDat_2024_06_20_PS2439_6_1K_dynamic.csv",
    # "ExpDat_2025_02_20_PS-A1000-dynamic-1K.csv",
    # "ExpDat_2025_02_20_005d-M1-dynamic-1K.csv",
    # "ExpDat_2025_02_20_PS-F1-dynamic-1K.csv",
    # "ExpDat_2025_02_20_005d-L1-dynamic-1K.csv",
]
dynamic_ms_filenames: list = [
    # "ExpDat_2024_06_20_005c-B_dynamic-aeolos.csv",
    # "ExpDat_2024_06_20_PS2439_6_1K_dynamic-aeolos.csv",
    # "ExpDat_2025_02_20_PS-A1000-dynamic-1K-aeolos.csv",
    # "ExpDat_2025_02_20_005d-M1-dynamic-1K-aeolos.csv",
    # "ExpDat_2025_02_20_PS-F1-dynamic-1K-aeolos.csv",
    # "ExpDat_2025_02_20_005d-L1-dynamic-1K-aeolos.csv",
]

# labels: list = ["PS-SCF3-1K_min", "PS-6.1K-1K_min"]
# labels: list = ["PS-SCF3", "PS-6.1K"]
# labels: list = ["PS-SCF3-0_1K_min", "PS-6_1K-0_1K_min"]
# labels: list = ["PS-1.12K", "PS-1.12K-SCF3"]
# labels: list = ["PS-10K", "PS-10K-SCF3"]
# labels: list = ["PS-19.6K", "PS-19.6K-SCF3"]
# labels: list = ["PS-110K", "PS-110K-SCF3"]

# labels: list = ["PS-SCF3-1K_min", "PS-6.1K-1K_min"]
labels: list = ["PS-10K", "PS-10K-SCF3"]
colors: list = [style["color"][labels[0]], style["color"][labels[1]]]
labels[1] = "PS-10K-SCF$_{3}$"

if __name__ == "__main__":
    isothermal_tga_plots = TGAPlots(
        data_dir=data_dir,
        tga_data_path=isothermal_filenames,
        ms_data_path=isothermal_ms_filenames,
        labels=labels,
        colors=colors,
        result_dir=result_path,
        style_path=style_path,
    )
    # TODO: must change temp and xlimit depending on the isothermal conditions!
    # isothermal_tga_plots.plot_tga_isothermal(
    #     isothermal_temp=300,
    #     xlim=(0, 1195),
    #     ylim=(0, 100),
    #     target_mass=104,
    #     initial_correction_time=60,
    #     uncertainty=False,
    #     time_for_mass_difference=1200,
    # )
    results = isothermal_tga_plots.compare_isothermal_degradation_rates(
        mass_loss_targets=[10, 20, 30, 40, 50, 60, 70, 80],  # % mass loss to compare
        initial_correction_time=60,  # Starting time (min)
        xlim=(0, 1195),
        ylim=(0, 100),
    )
    # isothermal_tga_plots.plot_tga_isothermal_rat0e_constant(
    #     isothermal_temp=300, initial_correction_time=60, fit_end_time=1200
    # )
    # isothermal_tga_plots.plot_tga_isothermal_kinetic_analysis(
    #     isothermal_temp=300, initial_correction_time=60, fit_end_time=1200
    # )
    # isothermal_tga_plots.plot_ms_peak_area(
    #     "isothermal", 300, initial_time=0, end_time=1500, m_z_start=60, m_z_end=150
    # )
    # dynamic_tga_plots = TGAPlots(
    #     data_dir=data_dir,
    #     tga_data_path=dynamic_filenames,
    #     ms_data_path=dynamic_ms_filenames,
    #     labels=labels,
    #     # colors=["#d9d9d9", "#a6bddb"], # 1.12K
    #     colors=["#bdbdbd", "#74a9cf"],  # 10K
    #     # colors=["#969696", "#3690c0"], # 19.6K
    #     # colors=["#636363", "#0570b0"], # 40.4K
    #     # colors=["#252525", "#045a8d"], # 110K
    #     result_dir=result_path,
    #     style_path=style_path,
    # )
    # dynamic_tga_plots.plot_tga_dynamic(
    #     t_depolymerization_cutoff=99.5,
    #     target_mass=104,
    #     xlim=(250, 400),
    #     initial_correction_temp=250,
    # )
