# Import all packages needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import json
from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from sklearn.metrics import r2_score
from matplotlib.ticker import ScalarFormatter
import pdb


# Import style.json
style_path: Path = Path.cwd() / "style" / "style.json"
with open(style_path, "r") as f:  # opens the style.json file
    style: dict = json.load(f)  # loads as a dictionary object

# Result path
result_path: Path = Path.cwd() / "results" / "Electrical" / "AL_1_35L" / "output_curves"

# Filenames
# TODO: add filenames and add corresponding (excel sheet) tabs into the dictionary
filenames: dict = {"2024_05_08_OC_AL_1_35L_2B_1.xls": ["OC_AL_1_35L_2B_1"]}


# Loading Data (data file must be in the same directory as jupyter notebook)
# Import data from excel file
# TODO: replace filename with the name of your data file
# TODO: import multiple data files and plot multiple files with correct naming.
def plot_output_curve(
    filenames: dict,
    color_order: list["str"] = [
        "green_blue_1",
        "green_blue_2",
        "green_blue_3",
        "green_blue_4",
        "green_blue_5",
        "green_blue_6",
        "green_blue_7",
        "green_blue_8",
        "green_blue_9",
    ],
):
    """Function that plots transfer curves of OFET."""
    replicate_mobility_data: dict = {}  # dictionary to store dataframes of replicates
    for filename in filenames.keys():
        # get root name of filename
        root_name: str = filename.split(".")[0]
        raw_data_filename: Path = (
            Path.cwd() / "data" / "Electrical" / "AL_1_35L" / filename
        )
        for tab_name in filenames[filename]:
            # Tell program to read the data
            raw_data: pd.DataFrame = pd.read_excel(
                raw_data_filename, tab_name
            )  # read data into a pandas dataframe
            # Find all the columns with DrainV(
            num_of_output_curves = 1
            for column in raw_data.columns:
                if "DrainV(" in column:
                    num_of_output_curves += 1
            fig, ax = plt.subplots(figsize=(6, 4))
            color_idx = 0
            for num in range(1, num_of_output_curves):
                gate_voltage_column_name: str = f"GateV({num})"
                ax.plot(
                    raw_data[f"DrainV({num})"],
                    -raw_data[f"DrainI({num})"],
                    label=f"$V_G$={raw_data[gate_voltage_column_name][0]}",
                    linewidth=1.5,
                    markersize=12,
                    color=style["color"][list(style["color"].keys())[color_idx]],
                )  # plot sqrt_DrainI
                color_idx += 1
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], loc="best", frameon=False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_ylabel("Drain Current, I$_D$ (A)")  # y-axis label
            ax.set_xlabel("Drain Voltage, V$_D$ (V)")  # x-axis label
            # y-axis and x-axis ticks
            ax.xaxis.set_minor_locator(
                AutoMinorLocator(2)
            )  # value of AutoMinorLocator dictates number of minor ticks between major ticks in X-axis
            ax.yaxis.set_minor_locator(
                AutoMinorLocator(2)
            )  # value of AutoMinorLocator dictates number of minor ticks between major ticks in Y-axis
            ax.tick_params(
                axis="y", direction="in"
            )  # direction of major ticks in Y-axis
            ax.tick_params(
                axis="x", direction="in"
            )  # direction of major ticks in X-axis
            ax.tick_params(
                axis="y", which="minor", length=4, direction="in"
            )  # direction and length of minor ticks in Y-axis
            ax.tick_params(
                axis="x", which="minor", length=4, direction="in"
            )  # direction and length of minor ticks in X-axis
            # Set scientific notation for ticks
            xfmt = ScalarFormatter()
            xfmt.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(xfmt)
            ax.yaxis.major.formatter._useMathText = True
            # NOTE: User can change the x and y limits here
            plt.ylim(
                0,
            )
            # Save the figure
            # NOTE: User can change the filename
            plt.savefig(
                result_path / f"OFET_output_curve_{root_name}.svg",
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                result_path / f"OFET_output_curve_{root_name}.jpg",
                dpi=300,
                bbox_inches="tight",
            )


if __name__ == "__main__":
    plot_output_curve(filenames)
    print("Output Curve template has been executed.")

print(plt.style.available)
