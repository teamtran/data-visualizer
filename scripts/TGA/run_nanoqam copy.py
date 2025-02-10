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
experiment_name: str = "005d-B"

# Result path
result_path: Path = root_path / "results" / "TGA" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "TGA" / experiment_name

# Filenames
isothermal_filenames: list = [
    "P2934-S TG.txt",
    "005dB1-rGPC+flash TG.txt",
    "005dB1-rGPC TG.txt",
    "005dB1-toyopearls TG.txt",
]
isothermal_ms_filenames: list = [
    "P2934-S MS.txt",
    "005dB1-rGPC+flash MS.txt",
    "005dB1-rGPC MS.txt",
    "005dB1-toyopearls MS.txt",
]


# labels: list = ["PS-SCF3-1K_min", "PS-6.1K-1K_min"]
labels: list = ["PS-6.1K", "PS-SCF3-FLASH", "PS-SCF3-rGPC", "PS-SCF3-Toyopearls"]
# labels: list = ["PS-SCF3-0_1K_min", "PS-6_1K-0_1K_min"]

if __name__ == "__main__":
    isothermal_tga_plots = TGAPlots(
        data_dir=data_dir,
        tga_data_path=isothermal_filenames,
        ms_data_path=isothermal_ms_filenames,
        labels=labels,
        colors=["#8286ff", "#f5c92a", "#e0f3db", "#7bccc4"],
        result_dir=result_path,
        style_path=style_path,
    )
    # TODO: must change temp and xlimit depending on the isothermal conditions!
    isothermal_tga_plots.plot_tga_isothermal(
        isothermal_temp=300,
        xlim=(0, 1450),
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
