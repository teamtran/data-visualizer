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
experiment_name: str = "commercial_samples"

# Result path
result_path: Path = root_path / "results" / "TGA" / experiment_name
if not result_path.exists():
    result_path.mkdir(parents=True)

# Data directory path
data_dir: Path = root_path / "data" / "TGA" / experiment_name

# Filenames
isothermal_filenames: list = [
    [
        "ExpDat_2025_02_28_Styrofoam_Isothermal.csv",
        "ExpDat_2025_07_12_Styrofoam_SCF3_isothermal.csv",
    ],
    [
        "ExpDat_2025_08_15_SL_PS_Container_Isothermal.csv",
        "ExpDat_2025_09_04_SL_PS_Container-SCF3_Isothermal.csv",
    ],
    [
        "ExpDat_2025_08_13_SL_PS_Cup_Lid_Isothermal.csv",
        "ExpDat_2025_08_21_SL_PS_Cup_Lid_SCF3_Isothermal.csv",
    ],
    [
        "ExpDat_2025_08_01_SL_PS_Petri_Dish_Isothermal.csv",
        "ExpDat_2025_09_14_SL_PS_Petri_Dish_SCF3_250912_P5EER7_V1_2_isothermal.csv",
    ],
    [
        "ExpDat_2025_07_25_SL_PS_Red_Solo_Cup_Isothermal.csv",
        "ExpDat_2025_08_24_SL_PS_Red_Solo_Cup_SCF3_Isothermal.csv",
    ],
    [
        "ExpDat_2025_08_03_SL_PS_Blue_Cup_Isothermal.csv",
        "ExpDat_2025_08_27_SL_PS_Blue_Cup_SCF3_Isothermal.csv",
    ],
]
isothermal_ms_filenames: list = [
    [
        "ExpDat_2025_02_28_Styrofoam_Isothermal-aeolos.csv",
        "ExpDat_2025_07_12_Styrofoam_SCF3_isothermal-aeolos.csv",
    ],
    [
        "ExpDat_2025_08_15_SL_PS_Container_Isothermal-aeolos.csv",
        "ExpDat_2025_09_04_SL_PS_Container-SCF3_Isothermal-aeolos.csv",
    ],
    [
        "ExpDat_2025_08_13_SL_PS_Cup_Lid_Isothermal-aeolos.csv",
        "ExpDat_2025_08_21_SL_PS_Cup_Lid_SCF3_Isothermal-aeolos.csv",
    ],
    [
        "ExpDat_2025_08_01_SL_PS_Petri_Dish_Isothermal-aeolos.csv",
        "ExpDat_2025_09_14_SL_PS_Petri_Dish_SCF3_250912_P5EER7_V1_2_isothermal-aeolos.csv",
    ],
    [
        "ExpDat_2025_07_25_SL_PS_Red_Solo_Cup_Isothermal-aeolos.csv",
        "ExpDat_2025_08_24_SL_PS_Red_Solo_Cup_SCF3_Isothermal-aeolos.csv",
    ],
    [
        "ExpDat_2025_08_03_SL_PS_Blue_Cup_Isothermal-aeolos.csv",
        "ExpDat_2025_08_27_SL_PS_Blue_Cup_SCF3_Isothermal-aeolos.csv",
    ],
]


labels: list = [
    ["PS-Styrofoam", "PS-Styrofoam-SCF3"],
    ["PS-Container", "PS-Container-SCF3"],
    ["PS-Coffee-Lid", "PS-Coffee-Lid-SCF3"],
    ["PS-Petri-Dish", "PS-Petri-Dish-SCF3"],
    ["PS-Red-Solo-Cup", "PS-Red-Solo-Cup-SCF3"],
    ["PS-Blue-Cup", "PS-Blue-Cup-SCF3"],
]
colors: list = ["#000000", "#ffcc02"]

if __name__ == "__main__":
    for filename, ms_filename, label in zip(
        isothermal_filenames, isothermal_ms_filenames, labels
    ):
        isothermal_tga_plots = TGAPlots(
            data_dir=data_dir,
            tga_data_path=filename,
            ms_data_path=ms_filename,
            labels=label,
            colors=colors,
            result_dir=result_path,
            style_path=style_path,
        )
        # TODO: must change temp and xlimit depending on the isothermal conditions!
        isothermal_tga_plots.plot_tga_isothermal(
            isothermal_temp=300,
            xlim=(0, 1200),
            ylim=(0, 100),
            target_mass=104,
            initial_correction_time=50,
            uncertainty=False,
            time_for_mass_difference=1200,
        )
        isothermal_tga_plots.plot_tga_isothermal_rate_constant(
            isothermal_temp=300, initial_correction_time=50, fit_end_time=1200
        )
