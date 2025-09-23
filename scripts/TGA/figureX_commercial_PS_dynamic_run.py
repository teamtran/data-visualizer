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
dynamic_filenames: list = [
    [
        "ExpDat_2025_03_20_Styrofoam_dynamic.csv",
        "ExpDat_2025_07_13_Styrofoam_SCF3_dynamic.csv",
    ],
    [
        "ExpDat_2025_08_17_SL_PS_Container_Dynamic.csv",
        "ExpDat_2025_09_08_SL_PS_Container-SCF3_Dynamic_P5EER5_V1_2.csv",
    ],
    [
        "ExpDat_2025_08_14_SL_PS_Cup_Lid_Dynamic.csv",
        "ExpDat_2025_08_22_SL_PS_Cup_Lid_SCF3_Dynamic.csv",
    ],
    [
        "ExpDat_2025_07_23_SL_PS_Petri_Dish_Dynamic.csv",
        "ExpDat_2025_09_15_SL_PS_Petri-Dish-SCF3_Dynamic_P5EER7_V1_2.csv",
    ],
    [
        "ExpDat_2025_07_22_SL_PS_Red_Solo_Cup_Dynamic.csv",
        "ExpDat_2025_09_07_SL_PS_Red_Solo_Cup_SCF3_P5EER3_V1_2_dynamic.csv",
    ],
    [
        "ExpDat_2025_08_14_SL_PS_Blue_Cup_Dynamic.csv",
        "ExpDat_2025_09_09_SL_PS_Blue_Cup_SCF3_P5EER3_V3_4_dynamic.csv",
    ],
]
dynamic_ms_filenames: list = [
    [
        "ExpDat_2025_03_20_Styrofoam_dynamic-aeolos.csv",
        "ExpDat_2025_07_13_Styrofoam_SCF3_dynamic-aeolos.csv",
    ],
    [
        "ExpDat_2025_08_17_SL_PS_Container_Dynamic-aeolos.csv",
        "ExpDat_2025_09_08_SL_PS_Container-SCF3_Dynamic_P5EER5_V1_2-aeolos.csv",
    ],
    [
        "ExpDat_2025_08_14_SL_PS_Cup_Lid_Dynamic-aeolos.csv",
        "ExpDat_2025_08_22_SL_PS_Cup_Lid_SCF3_Dynamic-aeolos.csv",
    ],
    [
        "ExpDat_2025_07_23_SL_PS_Petri_Dish_Dynamic-aeolos.csv",
        "ExpDat_2025_09_15_SL_PS_Petri-Dish-SCF3_Dynamic_P5EER7_V1_2-aeolos.csv",
    ],
    [
        "ExpDat_2025_07_22_SL_PS_Red_Solo_Cup_Dynamic-aeolos.csv",
        "ExpDat_2025_09_07_SL_PS_Red_Solo_Cup_SCF3_P5EER3_V1_2_dynamic-aeolos.csv",
    ],
    [
        "ExpDat_2025_08_14_SL_PS_Blue_Cup_Dynamic-aeolos.csv",
        "ExpDat_2025_09_09_SL_PS_Blue_Cup_SCF3_P5EER3_V3_4_dynamic-aeolos.csv",
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
colors = ["#000000", "#ffcc02"]

if __name__ == "__main__":
    for filename, ms_filename, label in zip(
        dynamic_filenames, dynamic_ms_filenames, labels
    ):
        dynamic_tga_plots = TGAPlots(
            data_dir=data_dir,
            tga_data_path=filename,
            ms_data_path=ms_filename,
            labels=label,
            colors=colors,
            result_dir=result_path,
            style_path=style_path,
        )
        dynamic_tga_plots.plot_tga_dynamic(
            t_depolymerization_cutoff=98,
            target_mass=104,
            xlim=(250, 400),
            initial_correction_temp=250,
        )
