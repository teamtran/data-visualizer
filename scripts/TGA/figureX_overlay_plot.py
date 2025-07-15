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

# isothermal
# isothermal_virgin_ps_data_paths: list = [
#     ("ExpDat_2025_07_05_SL_PS_Tosoh_A1000_isothermal.csv", "1.12K"),
#     ("ExpDat_2025_07_04_SL_PS_Tosoh_F1_isothermal.csv", "10K"),
#     ("ExpDat_2025_02_26_PS-F4-isothermal-1K.csv", "40.4K"),
# ]
isothermal_scf3_ps_data_paths: list = [
    ("ExpDat_2025_03_29_SL_005d_M1_isothermal.csv", "1.12K"),
    ("ExpDat_2025_03_14_SL_005D_L1_isothermal.csv", "10K"),
    ("ExpDat_2025_07_10_SL_PS_005d-L2_isothermal.csv", "19.6K"),
    # ("ExpDat_2025_02_25_005d-L3-isothermal-1K.csv", "40.4K"),
    ("ExpDat_2025_03_22_SL_005d_L4_isothermal.csv", "110K"),
]

# dynamic
dynamic_virgin_ps_data_paths: list = [
    ("ExpDat_2025_02_20_PS-A1000-dynamic-1K.csv", "1.12K"),
    ("ExpDat_2025_02_20_PS-F1-dynamic-1K.csv", "10K"),
    ("ExpDat_2025_02_21_PS-F2-dynamic-1K.csv", "19.6K"),
    ("ExpDat_2025_02_22_PS-F4-dynamic-1K.csv", "40.4K"),
    # F10
]
dynamic_scf3_ps_data_paths: list = [
    ("ExpDat_2025_02_20_005d-M1-dynamic-1K.csv", "1.12K"),
    ("ExpDat_2025_02_20_005d-L1-dynamic-1K.csv", "10K"),
    ("ExpDat_2025_02_22_005d-L2-dynamic-1K.csv", "19.6K"),
    ("ExpDat_2025_02_22_005d-L3-dynamic-1K.csv", "40.4K"),
    ("ExpDat_2025_02_27_005d-L4-dynamic-1K.csv", "110K"),
]

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
    tga_summary_plots.plot_overlay_isothermal(None, isothermal_scf3_ps_data_paths)
    tga_summary_plots.plot_overlay_dynamic(None, dynamic_scf3_ps_data_paths)
