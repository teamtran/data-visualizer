# Import all packages needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import scipy.stats as stats
import pandas as pd
import json
from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from sklearn.metrics import r2_score
from matplotlib.ticker import ScalarFormatter
import pdb

# TODO: Change global parameters HERE
# NOTE: change c, w, l
c: float = 1.15e-08  # (F/cm^2)
w: float = 0.1  # (cm)
l: float = 0.005  # (cm)


# Data preprocessing specific to OFET
def preprocess(data: pd.DataFrame, normalize: bool = False, baseline: bool = False):
    """Function that applies transformation to the dataframe which will make it ready for plotting."""
    # square root of absolute value of DrainI
    data["sqrt_DrainI"] = np.sqrt(np.abs(data["DrainI"]))

    return data


# When calculating the mobility, you want to calculate in the saturation regime
def calculate_mobility(slope_id_up: float, slope_id_down: float):
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
result_path: Path = (
    Path.cwd() / "results" / "Electrical" / "AL_1_33C" / "transfer_curves"
)

# Filenames
# TODO: add filenames and add corresponding (excel sheet) tabs into the dictionary
filenames: dict = {
    "2024_05_13_AL_1_33C_1A_D_all.xls": ["AL_1_33C_1D_1", "AL_1_33C_1D_2"]
}


# Loading Data
# Import data from excel file
# TODO: replace filename with the name of your data file
# TODO: import multiple data files and plot multiple files with correct naming
# TODO: add x range for each replicate
def plot_best_transfer_curves(
    filenames: dict,
    x0: int = -70,
    x1: int = -50,
    on_GateV: int = -80,
    off_GateV: int = -5,
):
    """Function that plots transfer curves of OFET."""
    replicate_mobility_data: dict = {}  # dictionary to store dataframes of replicates
    for filename in filenames.keys():
        # get root name of filename
        root_name: str = filename.split(".")[0]
        raw_data_filename: Path = (
            Path.cwd() / "data" / "Electrical" / "AL_1_33C" / filename
        )
        for tab_name in filenames[filename]:
            # Tell program to read the data
            raw_data: pd.DataFrame = pd.read_excel(
                raw_data_filename, tab_name
            )  # read data into a pandas dataframe

            # NOTE: preprocessing data to get sqrt_DrainI
            preprocessed_data: pd.DataFrame = preprocess(raw_data, True, True)

            # Segment the data into up curve and down curve
            # Look for index where in GateV column, there are two consecutives values that are equivalent
            # This is the index where the up curve ends and the down curve starts
            up_down_idx: int = preprocessed_data["GateV"].diff().eq(0).idxmax()

            # Plot transfer curves
            fig, ax = plt.subplots(figsize=(3, 4))  # create a figure and axis object
            ax2 = ax.twinx()  # create a second y-axis that shares the same x-axis
            ax.set_yscale("log")

            # plot DrainI and id_up
            ax.plot(
                preprocessed_data["GateV"][0:up_down_idx],
                np.abs(preprocessed_data["DrainI"][0:up_down_idx]),
                color=style["color"]["blue"],
                label="$\mathregular{I_{D}}$, up",
                linewidth=1.5,
                markersize=12,
            )
            # plot DrainI and id_down
            ax.plot(
                preprocessed_data["GateV"][up_down_idx:],
                np.abs(preprocessed_data["DrainI"][up_down_idx:]),
                color=style["color"]["yellow"],
                label="$\mathregular{I_{D}}$, down",
                linewidth=1.5,
                markersize=12,
            )

            ax.set_xlabel("Gate Voltage, $\mathregular{V_{G}}$ (V)")  # x-axis label
            ax.set_ylabel("-$\mathregular{I_{D}}$ (A)")  # y-axis label

            # Plot sqrt_DrainI for id_up
            ax2.plot(
                preprocessed_data["GateV"][0:up_down_idx],
                preprocessed_data["sqrt_DrainI"][0:up_down_idx],
                color=style["color"]["blue"],
                label="id_up",
                linewidth=1.5,
                markersize=12,
                linestyle="dotted",
            )
            # plot sqrt_DrainI for id_down
            ax2.plot(
                preprocessed_data["GateV"][up_down_idx:],
                preprocessed_data["sqrt_DrainI"][up_down_idx:],
                color=style["color"]["yellow"],
                label="id_down",
                linewidth=1.5,
                markersize=12,
                linestyle="dotted",
            )
            # find indicies of x values for up transfer curve
            idx1: int = preprocessed_data["GateV"][0:up_down_idx].sub(x0).abs().idxmin()
            idx0: int = preprocessed_data["GateV"][0:up_down_idx].sub(x1).abs().idxmin()

            # plot linear line that depicts the linear regime
            ax2.plot(
                [x0, x1],
                [
                    preprocessed_data["sqrt_DrainI"][idx1],
                    preprocessed_data["sqrt_DrainI"][idx0],
                ],
                color="red",
                label="linear_regime",
                linewidth=1.5,
                markersize=12,
                linestyle="--",
            )
            ax2.set_ylabel("√|$\mathregular{I_{D}}$| (√A)")  # y-axis label

            ax.legend(loc="upper right", frameon=False)  # legend for id_up and id_down

            # y-axis and x-axis ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
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

            ax2.tick_params(
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

            # NOTE: User can change the x and y limits here
            plt.xlim(-80, 20)

            # Despines the figure
            ax.spines["top"].set_visible(False)
            ax2.spines["top"].set_visible(False)

            # Save the figure
            # NOTE: User can change the filename
            plt.savefig(
                result_path / f"OFET_{tab_name}.svg", dpi=300, bbox_inches="tight"
            )
            plt.savefig(
                result_path / f"OFET_{tab_name}.jpg", dpi=300, bbox_inches="tight"
            )

            # Plot the linear fit zoomed in on the VG values used
            # For ID_up

            fig, ax = plt.subplots()  # create a figure and axis object

            # find indicies of x values
            idx0: int = preprocessed_data["GateV"][:up_down_idx].sub(x0).abs().idxmin()
            idx1: int = preprocessed_data["GateV"][:up_down_idx].sub(x1).abs().idxmin()
            idx1 += 1

            # plot sqrt_DrainI of saturation regime (up curve)
            ax.plot(
                preprocessed_data["GateV"][idx1:idx0],
                preprocessed_data["sqrt_DrainI"][idx1:idx0],
                color=style["color"]["blue"],
                label="sqrt_DrainI_up",
                linewidth=1.5,
                markersize=12,
            )  # plot sqrt_DrainI
            ax.set_ylabel("√|$\mathregular{I_{D}}$| (√A)")  # y-axis label
            ax.set_xlabel("Gate Voltage, V$_G$ (V)")  # x-axis label

            # calculate linear fit
            slope_id_up, intercept, r_value, p_value, std_err = stats.linregress(
                preprocessed_data["GateV"][idx1:idx0],
                preprocessed_data["sqrt_DrainI"][idx1:idx0],
            )
            print(slope_id_up, intercept, r_value, p_value, std_err)
            # get r2 of the linear fit
            r2: float = r2_score(
                preprocessed_data["sqrt_DrainI"][idx1:idx0],
                slope_id_up * preprocessed_data["GateV"][idx1:idx0] + intercept,
            )
            print(r2)
            # get Vthreshold by getting the x-intercept from the linear fit
            # y=mx+b, to get to x-intercept: y = 0, find x. -b/m = x
            vthreshold: float = -intercept / slope_id_up

            # plot linear fit
            ax.plot(
                preprocessed_data["GateV"][idx1:idx0],
                slope_id_up * preprocessed_data["GateV"][idx1:idx0] + intercept,
                color=style["color"]["dark_yellow"],
                label="linear_fit; slope: {:.7f}, intercept: {:.5f},\nR$^2$: {:.5f}".format(
                    slope_id_up, intercept, r2
                ),
                linewidth=1.5,
                markersize=12,
                linestyle=":",
            )

            # plot legend of slope, intercept, and R2
            ax.legend(loc="best", frameon=False)

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            plt.savefig(
                result_path / f"OFET_up_saturation_{tab_name}svg",
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                result_path / f"OFET_up_saturation_{tab_name}.jpg",
                dpi=300,
                bbox_inches="tight",
            )

            # Plot the linear fit zoomed in on the VG values used
            # For ID_down

            fig, ax = plt.subplots()  # create a figure and axis object

            # TODO: input x values for the saturation regime
            # find indicies of x values
            idx0: int = preprocessed_data["GateV"][up_down_idx:].sub(x0).abs().idxmin()
            idx1: int = preprocessed_data["GateV"][up_down_idx:].sub(x1).abs().idxmin()
            idx1 += 1

            # plot sqrt_DrainI of saturation regime (down curve)
            ax.plot(
                preprocessed_data["GateV"][idx0:idx1],
                preprocessed_data["sqrt_DrainI"][idx0:idx1],
                color=style["color"]["yellow"],
                label="sqrt_DrainI_down",
                linewidth=1.5,
                markersize=12,
            )  # plot sqrt_DrainI
            ax.set_ylabel("√|$\mathregular{I_{D}}$| (√A)")  # y-axis label
            ax.set_xlabel("Gate Voltage, V$_G$ (V)")  # x-axis label

            # calculate the linear fit
            slope_id_down, intercept, r_value, p_value, std_err = stats.linregress(
                preprocessed_data["GateV"][idx0:idx1],
                preprocessed_data["sqrt_DrainI"][idx0:idx1],
            )
            print(slope_id_down, intercept, r_value, p_value, std_err)
            # get r2 of the linear fit
            r2 = r2_score(
                preprocessed_data["sqrt_DrainI"][idx0:idx1],
                slope_id_down * preprocessed_data["GateV"][idx0:idx1] + intercept,
            )
            print(r2)
            # plot linear fit
            ax.plot(
                preprocessed_data["GateV"][idx0:idx1],
                slope_id_down * preprocessed_data["GateV"][idx0:idx1] + intercept,
                color=style["color"]["dark_blue"],
                label="linear_fit; slope: {:.7f}, intercept: {:.5f}, \nR$^2$: {:.5f}".format(
                    slope_id_down, intercept, r2
                ),
                linewidth=1.5,
                markersize=12,
                linestyle=":",
            )
            # plot legend of slope, intercept, and R2
            ax.legend(loc="best", frameon=False)

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            plt.savefig(
                result_path / f"OFET_down_saturation_{tab_name}.svg",
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                result_path / f"OFET_down_saturation_{tab_name}.jpg",
                dpi=300,
                bbox_inches="tight",
            )
            # calculate mobility
            u_down, u_up, u_average = calculate_mobility(slope_id_up, slope_id_down)

            # Report ID on and off (for up curve only)
            # Calculate the ID on/off

            id_on_idx: int = (
                preprocessed_data["GateV"][:up_down_idx].sub(on_GateV).abs().idxmin()
            )
            id_off_idx: int = (
                preprocessed_data["GateV"][:up_down_idx].sub(off_GateV).abs().idxmin()
            )
            id_on: float = preprocessed_data["DrainI"][id_on_idx]
            id_off: float = preprocessed_data["DrainI"][id_off_idx]
            id_on_off: float = id_on / id_off

            # column names for data in csv file

            replicate_mobility_data[tab_name] = [
                u_down,
                u_up,
                u_average,
                id_on,
                id_off,
                id_on_off,
                vthreshold,
            ]

    # # calculate average mobility
    mobility_data: pd.DataFrame = pd.DataFrame.from_dict(
        replicate_mobility_data,
        orient="index",
        columns=[
            "u_down",
            "u_up",
            "u_average",
            "id_on",
            "id_off",
            "id_on_off",
            "vthreshold",
        ],
    )

    # average each column individually
    u_down_across_replicates: float = mobility_data["u_down"].mean()
    u_up_across_replicates: float = mobility_data["u_up"].mean()
    u_average_across_replicates: float = mobility_data["u_average"].mean()
    id_on_average_across_replicates: float = mobility_data["id_on"].mean()
    id_off_average_across_replicates: float = mobility_data["id_off"].mean()
    id_on_off_average_across_replicates: float = mobility_data["id_on_off"].mean()
    vthreshold_average: float = mobility_data["vthreshold"].mean()

    # standard deviation of each column
    u_down_std: float = mobility_data["u_down"].std()
    u_up_std: float = mobility_data["u_up"].std()
    n: int = len(mobility_data)
    u_average_std: float = np.sqrt(
        (n * u_down_std**2 + n * u_up_std**2) / (n + n)
    )  # https://www.geeksforgeeks.org/combined-standard-deviation-meaning-formula-and-example/
    # add a row to the dataframe with the average mobility
    id_on_std: float = mobility_data["id_on"].std()
    id_off_std: float = mobility_data["id_off"].std()
    id_on_off_std: float = mobility_data["id_on_off"].std()
    vthreshold_std: float = mobility_data["vthreshold"].std()

    mobility_data.loc["Average"] = [
        u_down_across_replicates,
        u_up_across_replicates,
        u_average_across_replicates,
        id_on_average_across_replicates,
        id_off_average_across_replicates,
        id_on_off_average_across_replicates,
        vthreshold_average,
    ]
    mobility_data.loc["Std"] = [
        u_down_std,
        u_up_std,
        u_average_std,
        id_on_std,
        id_off_std,
        id_on_off_std,
        vthreshold_std,
    ]
    mobility_data.to_csv(result_path / f"{root_name}_mobility_data.csv")


if __name__ == "__main__":
    plot_best_transfer_curves(filenames)
    print("OFET template has been executed.")
