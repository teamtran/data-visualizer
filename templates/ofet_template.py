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

# TODO: Change global parameters HERE
# NOTE: change c, w, l
c: float = 1.15e-08  # (F/cm^2)
w: float = 0.1  # (cm)
l: float = 0.005  # (cm)


# Data preprocessing specific to OFET
def preprocess(data: pd.DataFrame, normalize: bool = False, baseline: bool = False):
    """Function that applies transformation to the dataframe which will make it ready for plotting. Note, this is specific to UV-Vis."""
    # square root of absolute value of id_down and id_up
    data["sqrt_id_down"] = np.sqrt(np.abs(data["id_down"]))
    data["sqrt_id_up"] = np.sqrt(np.abs(data["id_up"]))

    return data


# When calculating the mobility, you will want to look at output curve (gradient ~ 0) and then draw the linear saturation regime from there. In addition, you will want to do triplicates.
def calculate_mobility(slope_id_down: float, slope_id_up: float):
    """Function that calculates the mobility of the OFET."""
    u_down: float = (2 * l * (np.abs(slope_id_down) ** 2)) / (c * w)
    u_up: float = (2 * l * (np.abs(slope_id_up) ** 2)) / (c * w)

    u_average: float = np.mean([u_down, u_up])
    print(f"{u_down=}, {u_up=}, {u_average=}")
    return u_down, u_up, u_average


# Import style.json
style_path: Path = Path.cwd() / "style" / "style.json"
with open(style_path, "r") as f:  # opens the style.json file
    style: dict = json.load(f)  # loads as a dictionary object

# Result path
result_path: Path = Path.cwd() / "results"

# Filenames
# TODO: add more filenames if you have more replicates
filenames: list = [
    "transfer_curve_AL_1_36L_1.txt",
    "transfer_curve_AL_1_36L_2.txt",
    "transfer_curve_AL_1_36L_3.txt",
    "transfer_curve_AL_1_36L_4.txt",
]

# TODO: add x range for linear regime for each replicate and Id_up and Id_down


# Loading Data (data file must be in the same directory as jupyter notebook)
# Import data from excel file
# TODO: replace filename with the name of your data file
# TODO: import multiple data files and plot multiple files with correct naming.
def plot_transfer_curves(filenames: list):
    """Function that plots transfer curves of OFET."""
    replicate_mobility_data: dict = {}  # dictionary to store dataframes of replicates
    for filename in filenames:
        # get root name of filename
        root_name: str = filename.split(".")[0]
        raw_data_filename: Path = Path.cwd() / "data" / filename

        # Tell program to read the data
        raw_data: pd.DataFrame = pd.read_csv(
            raw_data_filename, sep="	"
        )  # read data into a pandas dataframe

        # NOTE: change True to False if you don't want normalization or baseline correction
        preprocessed_data: pd.DataFrame = preprocess(raw_data, True, True)

        # Plot transfer curves
        fig, ax = plt.subplots()  # create a figure and axis object
        ax2 = ax.twinx()  # create a second y-axis that shares the same x-axis

        # plot id_down and id_up
        ax.plot(
            preprocessed_data["vg"],
            -preprocessed_data["id_up"],
            color=style["color"]["blue"],
            label="$\mathregular{I_{D}}$, up",
            linewidth=2,
            markersize=12,
        )  # plot id_up
        ax.plot(
            preprocessed_data["vg"],
            -preprocessed_data["id_down"],
            color=style["color"]["yellow"],
            label="$\mathregular{I_{D}}$, down",
            linewidth=2,
            markersize=12,
        )  # plot id_down
        ax.set_yscale("log")
        ax.set_xlabel("Gate Voltage, $\mathregular{V_{G}}$ (V)")  # x-axis label
        ax.set_ylabel("-$\mathregular{I_{D}}$ (A)")  # y-axis label

        # Plot sqrt_id_down and sqrt_id_up
        ax2.plot(
            preprocessed_data["vg"],
            preprocessed_data["sqrt_id_up"],
            color=style["color"]["blue"],
            label="id_up",
            linewidth=2,
            markersize=12,
            linestyle="dashed",
        )  # plot id_down
        ax2.plot(
            preprocessed_data["vg"],
            preprocessed_data["sqrt_id_down"],
            color=style["color"]["yellow"],
            label="id_down",
            linewidth=2,
            markersize=12,
            linestyle="dashed",
        )  # plot id_down
        ax2.set_ylabel("√|$\mathregular{I_{D}}$| (√A)")  # y-axis label

        ax.legend(loc="upper right")  # legend for id_down and id_up

        # y-axis and x-axis ticks
        ax.xaxis.set_minor_locator(
            AutoMinorLocator(2)
        )  # value of AutoMinorLocator dictates number of minor ticks between major ticks in X-axis
        ax.yaxis.set_minor_locator(
            AutoMinorLocator(2)
        )  # value of AutoMinorLocator dictates number of minor ticks between major ticks in Y-axis
        ax.tick_params(axis="y", direction="in")  # direction of major ticks in Y-axis
        ax.tick_params(axis="x", direction="in")  # direction of major ticks in X-axis
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
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-3, 3))
        ax2.yaxis.set_major_formatter(xfmt)
        ax2.yaxis.major.formatter._useMathText = True
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        # NOTE: User can change the x and y limits here
        plt.xlim(-60, 5)

        # NOTE: make sure you comment this out if you want to save your figure in the directory. Otherwise, your figure will look like a blank screen
        # plt.show()

        # Despines the figure
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        # Save the figure
        # NOTE: User can change the filename
        plt.savefig(result_path / f"OFET_{root_name}.svg", dpi=300, bbox_inches="tight")
        plt.savefig(result_path / f"OFET_{root_name}.jpg", dpi=300, bbox_inches="tight")

        # Plot ID down saturation regimes of transfer curves
        # Plot the fitted curve
        fig, ax = plt.subplots()  # create a figure and axis object

        # TODO: input x values for the saturation regime
        x0: float = -50
        x1: float = -35
        # find indicies of x values
        idx0: int = preprocessed_data["vg"].sub(x0).abs().idxmin()
        idx1: int = preprocessed_data["vg"].sub(x1).abs().idxmin()

        # plot sqrt_id_down of linear saturation regime
        ax.plot(
            preprocessed_data["vg"][idx0:idx1],
            preprocessed_data["sqrt_id_down"][idx0:idx1],
            color=style["color"]["yellow"],
            label="id_down",
            linewidth=2,
            markersize=12,
            linestyle="dashed",
        )  # plot sqrt_id_down
        ax.set_ylabel("√|$\mathregular{I_{D}}$| (√A)")  # y-axis label
        ax.set_xlabel("Gate Voltage, V$_G$ (V)")  # x-axis label

        # calculate the fitted curve
        slope_id_down, intercept, r_value, p_value, std_err = stats.linregress(
            preprocessed_data["vg"][idx0:idx1],
            preprocessed_data["sqrt_id_down"][idx0:idx1],
        )
        print(slope_id_down, intercept, r_value, p_value, std_err)
        # get r2 of the fitted curve
        r2 = r2_score(
            preprocessed_data["sqrt_id_down"][idx0:idx1],
            slope_id_down * preprocessed_data["vg"][idx0:idx1] + intercept,
        )
        print(r2)
        # plot fitted curve
        ax.plot(
            preprocessed_data["vg"][idx0:idx1],
            slope_id_down * preprocessed_data["vg"][idx0:idx1] + intercept,
            color=style["color"]["dark_yellow"],
            label="Fitted Curve",
            linewidth=2,
            markersize=12,
            linestyle=":",
        )  # plot fitted curve
        # plot text box of slope, intercept, and R2
        ax.text(x0 + 1, 0.0006, "Slope: {:.5e}".format(slope_id_down), fontsize=12)
        ax.text(x0 + 1, 0.00055, "Intercept: {:.5e}".format(intercept), fontsize=12)
        ax.text(x0 + 1, 0.0005, "R$^2$: {:.5f}".format(r2), fontsize=12)

        # set y-axis ticks
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(xfmt)
        ax.yaxis.major.formatter._useMathText = True

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.savefig(
            result_path / f"OFET_down_saturation_{root_name}.svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            result_path / f"OFET_down_saturation_{root_name}.jpg",
            dpi=300,
            bbox_inches="tight",
        )

        # Plot ID up saturation regimes of transfer curves
        # Plot the fitted curve
        fig, ax = plt.subplots()  # create a figure and axis object

        # TODO: input x values for the saturation regime
        x0: float = -60
        x1: float = -40
        # find indicies of x values
        idx0: int = preprocessed_data["vg"].sub(x0).abs().idxmin()
        idx1: int = preprocessed_data["vg"].sub(x1).abs().idxmin()

        # plot sqrt_id_down of linear saturation regime
        ax.plot(
            preprocessed_data["vg"][idx0:idx1],
            preprocessed_data["sqrt_id_up"][idx0:idx1],
            color=style["color"]["blue"],
            label="id_up",
            linewidth=2,
            markersize=12,
            linestyle="dashed",
        )  # plot sqrt_id_down
        ax.set_ylabel("√|$\mathregular{I_{D}}$| (√A)")  # y-axis label
        ax.set_xlabel("Gate Voltage, V$_G$ (V)")  # x-axis label

        # calculate the fitted curve
        slope_id_up, intercept, r_value, p_value, std_err = stats.linregress(
            preprocessed_data["vg"][idx0:idx1],
            preprocessed_data["sqrt_id_up"][idx0:idx1],
        )
        print(slope_id_up, intercept, r_value, p_value, std_err)
        # get r2 of the fitted curve
        r2 = r2_score(
            preprocessed_data["sqrt_id_up"][idx0:idx1],
            slope_id_up * preprocessed_data["vg"][idx0:idx1] + intercept,
        )
        print(r2)
        # plot fitted curve
        ax.plot(
            preprocessed_data["vg"][idx0:idx1],
            slope_id_up * preprocessed_data["vg"][idx0:idx1] + intercept,
            color=style["color"]["dark_blue"],
            label="Fitted Curve",
            linewidth=2,
            markersize=12,
            linestyle=":",
        )  # plot fitted curve
        # plot text box of slope, intercept, and R2
        ax.text(x0 + 1, 0.0006, "Slope: {:.5e}".format(slope_id_up), fontsize=12)
        ax.text(x0 + 1, 0.00055, "Intercept: {:.5e}".format(intercept), fontsize=12)
        ax.text(x0 + 1, 0.0005, "R$^2$: {:.5f}".format(r2), fontsize=12)

        # set y-axis ticks
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(xfmt)
        ax.yaxis.major.formatter._useMathText = True

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.savefig(
            result_path / f"OFET_up_saturation_{root_name}.svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            result_path / f"OFET_up_saturation_{root_name}.jpg",
            dpi=300,
            bbox_inches="tight",
        )
        # calculate mobility
        u_down, u_up, u_average = calculate_mobility(slope_id_down, slope_id_up)
        replicate_mobility_data[root_name] = [u_down, u_up, u_average]
    # calculate average mobility
    mobility_data: pd.DataFrame = pd.DataFrame.from_dict(
        replicate_mobility_data, orient="index", columns=["u_down", "u_up", "u_average"]
    )
    # average each column individually
    u_down_avg_across_replicates: float = mobility_data["u_down"].mean()
    u_up_avg_across_replicates: float = mobility_data["u_up"].mean()
    u_average_avg_across_replicates: float = mobility_data["u_average"].mean()
    # add a row to the dataframe with the average mobility
    mobility_data.loc["Average"] = [
        u_down_avg_across_replicates,
        u_up_avg_across_replicates,
        u_average_avg_across_replicates,
    ]
    sample_name: str = (
        filename.split("_")[0]
        + "_"
        + filename.split("_")[1]
        + "_"
        + filename.split("_")[2]
        + "_"
        + filename.split("_")[3]
        + "_"
        + filename.split("_")[4]
    )
    mobility_data.to_csv(result_path / f"{sample_name}_mobility_data.csv")


if __name__ == "__main__":
    plot_transfer_curves(filenames)
    print("OFET template has been executed.")
