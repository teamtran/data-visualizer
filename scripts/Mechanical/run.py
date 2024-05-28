from pathlib import Path
import sys
from matplotlib import pyplot as plt
import json

# import pdb

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))
from code.Mechanical.mechanical_plots import MechanicalPlot
from code.data_util import get_filepath_auto


# Style path
style_path: Path = root_path / "style" / "style.json"
# read style file
style: dict = json.loads(style_path.read_text())
# defines font, fontsize, and axes linewidth of all of the following plots
plt.rcParams["font.family"] = style["fontfamily"]
plt.rcParams["font.size"] = style["fontsize"]
plt.rcParams["axes.linewidth"] = style["axes_linewidth"]


# Result path
tensile_result_path: Path = root_path / "results" / "Mechanical" / "Tensile"
compression_result_path: Path = root_path / "results" / "Mechanical" / "Compression"

# Data directory path
data_dir: Path = root_path / "data" / "Mechanical"

filename: str = "2024_03_28 013.A.6 4k10percent 30minvac  Test003Data.csv"
cycle_filename: str = (
    "2024_05_03_013_A_9_8k10p_rate cycle 0.01 to 5mm s-1 at 25p Test005Data.csv"
)
compression_filename: str = "2024_04_12 013.A.7 8k10percent 4h vac Test023Data.csv"

if __name__ == "__main__":
    # filename path
    data_path: Path = data_dir / filename
    # Tensile Plot
    plot = MechanicalPlot(
        data_path=data_path,
        result_path=tensile_result_path,
        style_path=style_path,
        cycle_label="1-Stretch",
        test_type="tensile",
    )
    plot._preprocess_data()
    plot.plot_curve(linear_regime_threshold=0.01, linear_regime_sampling_rate=5)

    # Cycle Tensile Plots
    cycle_data_path: Path = data_dir / cycle_filename
    cycle_plot = MechanicalPlot(
        data_path=cycle_data_path,
        result_path=tensile_result_path,
        style_path=style_path,
        cycle_label="1-Stretch",
        test_type="tensile",
    )
    cycle_plot._preprocess_data()
    cycle_plot.plot_cycle_curves()

    # Compression Plots
    compression_data_path: Path = data_dir / compression_filename
    compression_plot = MechanicalPlot(
        data_path=compression_data_path,
        result_path=compression_result_path,
        style_path=style_path,
        cycle_label="1-Compress",
        test_type="compression",
    )
    compression_plot._preprocess_data()
    compression_plot.plot_curve(
        linear_regime_threshold=0.2, linear_regime_sampling_rate=5
    )
