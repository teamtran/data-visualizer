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
experiment_name: str = "ps_scf3_flynn_ozawa_wall"

# Result path
result_path: Path = root_path / "results" / "TGA" / experiment_name
if not result_path.exists():
    result_path.mkdir(parents=True)

# Data directory path
data_dir: Path = root_path / "data" / "TGA" / experiment_name


dynamic_filenames: list = [
    "ExpDat_2025_11_18_SL_PS_Tosoh_F1_1K_min.csv",
    "ExpDat_2025_11_18_SL_PS_Tosoh_F1_2K_min.csv",
    "ExpDat_2025_11_18_SL_PS_Tosoh_F1_5K_min.csv",
    "ExpDat_2025_11_18_SL_PS_Tosoh_F1_10K_min.csv",
    # "ExpDat_2025_11_18_SL_PS_005D-L1_1K_min.csv",
    # "ExpDat_2025_11_18_SL_PS_005D-L1_2K_min.csv",
    # "ExpDat_2025_11_18_SL_PS_005D-L1_5K_min.csv",
    # "ExpDat_2025_11_18_SL_PS_005D-L1_10K_min.csv",
]
dynamic_ms_filenames: list = [
    "ExpDat_2025_11_18_SL_PS_Tosoh_F1_1K_min-aeolos.csv",
    "ExpDat_2025_11_18_SL_PS_Tosoh_F1_2K_min-aeolos.csv",
    "ExpDat_2025_11_18_SL_PS_Tosoh_F1_5K_min-aeolos.csv",
    "ExpDat_2025_11_18_SL_PS_Tosoh_F1_10K_min-aeolos.csv",
    # "ExpDat_2025_11_18_SL_PS_005D-L1_1K_min-aeolos.csv",
    # "ExpDat_2025_11_18_SL_PS_005D-L1_2K_min-aeolos.csv",
    # "ExpDat_2025_11_18_SL_PS_005D-L1_5K_min-aeolos.csv",
    # "ExpDat_2025_11_18_SL_PS_005D-L1_10K_min-aeolos.csv",
]

labels: list = [
    "PS_F1_1K_min",
    "PS_F1_2K_min",
    "PS_F1_5K_min",
    "PS_F1_10K_min",
    # "PSSCF3_1K_min",
    # "PSSCF3_2K_min",
    # "PSSCF3_5K_min",
    # "PSSCF3_10K_min",
]

colors: list = ["#000000", "#000000", "#000000", "#000000"]

if __name__ == "__main__":
    dynamic_tga_plots = TGAPlots(
        data_dir=data_dir,
        tga_data_path=dynamic_filenames,
        ms_data_path=dynamic_ms_filenames,
        labels=labels,
        colors=colors,
        result_dir=result_path,
        style_path=style_path,
    )
    dynamic_tga_plots.compare_flynn_ozawa_wall_experiments(
        [
            Path(
                "PS_F1_1K_min_PS_F1_2K_min_PS_F1_5K_min_PS_F1_10K_min_fow_astm_data.csv"
            ),
            Path(
                "PSSCF3_1K_min_PSSCF3_2K_min_PSSCF3_5K_min_PSSCF3_10K_min_fow_astm_data.csv"
            ),
        ],
        labels=["PS-10K", "PS-10K-SCF3"],
        colors=["#000000", "#ffcc02"],
        ylim=(300, 450),
        xlim=(0, 20),
    )
