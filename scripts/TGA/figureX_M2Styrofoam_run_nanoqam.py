from pathlib import Path
import sys
from matplotlib import pyplot as plt
import json

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))

from code.TGA.tga_plots_nanoqam import TGAPlots

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

# Data directory path
data_dir: Path = root_path / "data" / "TGA" / experiment_name

# Filenames
isothermal_filenames: list = [
    "styrofoam-corrected.txt",
    "M2.txt",
]
isothermal_ms_filenames: list = [
    "styrofoam_000001 -temp.txt",
    "M2_000001 -temp.txt",
]


# labels: list = ["PS-SCF3-1K_min", "PS-6.1K-1K_min"]
labels: list = ["PS-Styrofoam", "PS-Styrofoam-SCF3"]
colors: list = [style["color"][labels[0]], style["color"][labels[1]]]
labels[1] = "PS-Styrofoam-SCF$_{3}$"

# labels: list = ["PS-SCF3-0_1K_min", "PS-6_1K-0_1K_min"]

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
    isothermal_tga_plots.plot_tga_isothermal(
        isothermal_temp=300,
        xlim=(0, 1250),
        ylim=(0, 110),
        target_mass=104,
        initial_correction_time=50,
        uncertainty=False,
        time_for_mass_difference=1250,
    )
    # isothermal_tga_plots.plot_ms_peak_area(
    #     "isothermal", 300, initial_time=0, end_time=1500, m_z_start=60, m_z_end=150
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
    #     xlim=(230, 350),
    #     initial_correction_temp=230,
    # )
