from pathlib import Path
import sys

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))
from code.Electrical.electrical_plots import (
    TransferCurve,
    OutputCurve,
    OverlayTransferCurves,
    MW_Curves,
)
from code.data_util import get_filepath_auto


# Style path
style_path: Path = root_path / "style" / "style.json"

# TODO: Change experiment name and filenames
# Transfer curves
# Experiment Name
experiment_name: str = "AL_1_35L"

# Result path
result_path: Path = root_path / "results" / "Electrical" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "Electrical" / experiment_name

# Filenames
# TODO: add filenames and add corresponding (excel sheet) tabs into the dictionary (keys for filename, and values for sheet names)
transfer_curve_data_filenames: dict = {
    "2024_05_07_AL_1_35L_1A_3.xls": ["AL_1_35L_1A_3"]
}

# Output curve filename
output_curve_data_filenames: dict = {
    "2024_05_07_OC_AL_1_35L_1A_1.xls": ["OC_AL_1_35L_1A_1"]
}

# Overlay data directory path
overlay_data_dir: Path = root_path / "data" / "Electrical"

# Overlay result directory path
overlay_result_dir: Path = root_path / "results" / "Electrical"
# Overlay curve filenames
# 1 sheet per line in the dictionary (item is not a list but a string)
overlay_transfer_curve_data_filenames: dict = {
    "AL_1_35L/2024_05_07_AL_1_35L_1A_3.xls": "AL_1_35L_1A_3",
    "AL_1_33C/2024_05_13_AL_1_33C_1A_D_all.xls": "AL_1_33C_1D_1",
}

# Molecular Weight Data
mw_data_filepath: Path = root_path / "data" / "Electrical" / "MW" / "MW_data.csv"

# Molecular Weight Result Path
mw_result_path: Path = root_path / "results" / "Electrical" / "MW"

if __name__ == "__main__":
    # Transfer Curves
    # loop over all
    for data_filename in transfer_curve_data_filenames.keys():
        # Get the data file path
        data_filepath = data_dir / data_filename
        # iterate over sheet_names in the dictionary
        for sheet_name in transfer_curve_data_filenames[data_filename]:
            # Create the plot object
            plot = TransferCurve(
                data_filepath,
                sheet_name,
                result_path,
                style_path,
                c=1.15e-08,
                w=0.1,
                l=0.005,
                up_color="blue",
                down_color="yellow",
            )
            # Plot transfer curve
            plot.plot_transfer_curve(
                xlim=(-80, 0), max_x_range=-40, min_x_range=-60, direction="both"
            )
            # Plot saturation regime curves
            plot.plot_up_saturation_regime(max_x_range=-40, min_x_range=-60)
            plot.plot_down_saturation_regime(max_x_range=-40, min_x_range=-60)
            # Export metadata (must plot saturation regime curves first)
            plot.export_metadata(on_GateV=-60, off_GateV=-40)

    # Output Curves
    for data_filename in output_curve_data_filenames.keys():
        # Get the data file path
        data_filepath = data_dir / data_filename
        # iterate over sheet_names in the dictionary
        for sheet_name in output_curve_data_filenames[data_filename]:
            # Create the plot object
            plot = OutputCurve(data_filepath, sheet_name, result_path, style_path)
            # Plot output curve
            plot.plot_output_curve(
                color_order=[
                    "green_blue_1",
                    "green_blue_2",
                    "green_blue_3",
                    "green_blue_4",
                    "green_blue_5",
                    "green_blue_6",
                    "green_blue_7",
                    "green_blue_8",
                    "green_blue_9",
                    "green_blue_10",
                ]
            )

    # Overlay Transfer Curves
    # NOTE: specify name of overlay transfer curves
    overlay_transfer_curve_name: str = "IDT-BT_IDT-TPD"
    plot = OverlayTransferCurves(
        overlay_transfer_curve_data_filenames,
        overlay_data_dir,
        overlay_result_dir,
        overlay_transfer_curve_name,
        style_path,
    )
    plot.plot_overlay_transfer_curve(
        ["IDT-BT", "IDT-TPD"], color_order=["blue", "yellow"]
    )

    # Molecular Weight Plots
    # plot = MW_Curves(mw_data_filepath, mw_result_path, style_path)
    # plot.plot_mobility_vs_mw("blue", "IDTBT")
    # plot.plot_modulus_cos_strain_vs_mw("red", "green", "yellow", "IDTBT")
