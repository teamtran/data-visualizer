from pathlib import Path
import sys

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))
from code.Electrical.electrical_plots import ElectricalPlot
from code.data_util import get_filepath_auto

# Result path
result_path: Path = (
    root_path / "results" / "Electrical" / "AL_1_33C" / "transfer_curves"
)

# Data directory path
data_dir: Path = root_path / "data" / "Electrical" / "AL_1_33C"

# Style path
style_path: Path = root_path / "style" / "style.json"

# Filenames
# TODO: add filenames and add corresponding (excel sheet) tabs into the dictionary
data_filenames: dict = {
    "2024_05_13_AL_1_33C_1A_D_all.xls": ["AL_1_33C_1D_1", "AL_1_33C_1D_2"]
}

if __name__ == "__main__":
    # loop over all
    for data_filename in data_filenames.keys():
        # Get the data file path
        data_filepath = data_dir / data_filename
        # iterate over sheet_names in the dictionary
        for sheet_name in data_filenames[data_filename]:
            # Create the plot object
            plot = ElectricalPlot(
                data_filepath, sheet_name, result_path, style_path, 1.15e-08, 0.1, 0.005
            )
            # Plot transfer curve
            plot.plot_transfer_curve(auto_linear_regime=True)
            # Plot saturation regime curves
            plot.plot_up_saturation_regime()
            plot.plot_down_saturation_regime()
            # Export metadata
            plot.export_metadata()
