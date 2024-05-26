import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import scipy.stats as stats
import pandas as pd
import json
import os
from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from sklearn.metrics import r2_score
from matplotlib.ticker import ScalarFormatter
import pdb


class TransferCurve:
    """
    Class that contains methods to plot electrical data.
    """

    def __init__(
        self,
        data_path: Path,
        sheet_name: str,
        result_dir: Path,
        style_path: Path,
        c: float,
        w: float,
        l: float,
        up_color: str = "blue",
        down_color: str = "yellow",
    ):
        """Initialize class with data directory, result directory, c, w, and l.

        Args:
            data_dir (Path): path file to .csv with data
            result_dir (Path): result directory to save plots
            c (float): _description_ (F/cm^2)
            w (float): channel width (cm)
            l (float): channel length (cm)
        """
        self.data_dir = data_path
        self.sheet_name = sheet_name
        self.root_name = self.data_dir.stem
        self.result_dir = result_dir
        self.c = c
        self.w = w
        self.l = l
        with open(style_path, "r") as f:  # opens the style.json file
            self.style: dict = json.load(f)  # loads as a dictionary object
        self._preprocess_data()
        self._find_up_down_idx()
        self.up_color = self.style["color"][up_color]
        self.down_color = self.style["color"][down_color]

        # if directory is not present, make the directory
        if not os.path.isdir(self.result_dir / self.root_name):
            os.makedirs(self.result_dir / self.root_name)

    def _preprocess_data(self):
        """
        Preprocess the data for plotting.
        """
        # load the .xls into a pandas dataframe
        self.data = pd.read_excel(self.data_dir, self.sheet_name)
        # square root the absolute value of DrainI to form a new column
        self.data["sqrt_DrainI"] = np.sqrt(np.abs(self.data["DrainI"]))

    def _find_up_down_idx(self):
        """
        Segment the data into up curve and down curve
        Look for index where in GateV column, there are two consecutives values that are equivalent
        This is the index where the up curve ends and the down curve starts
        """
        self.up_down_idx: int = self.data["GateV"].diff().eq(0).idxmax()

    def _calculate_mobility(self, slope_id_up: float, slope_id_down: float):
        """
        Function that calculates the mobility of the OFET.

        Args:
            slope_id_up (float): slope of the up saturation regime
            slope_id_down (float): slope of the down saturation regime
        """
        u_down: float = (2 * self.l * (slope_id_down**2)) / (self.c * self.w)
        u_up: float = (2 * self.l * (slope_id_up**2)) / (self.c * self.w)

        u_average: float = np.mean([u_down, u_up])
        return u_down, u_up, u_average

    def _calculate_id_on_off_ratio(self, on_GateV: float, off_GateV: float):
        """
        Function that calculates the on/off ratio of the OFET.

        Args:
            slope_id_up (float): slope of the up saturation regime
            slope_id_down (float): slope of the down saturation regime
        """
        id_on_idx: int = self._find_index_given_value(
            self.data["GateV"][: self.up_down_idx], on_GateV
        )
        id_off_idx: int = self._find_index_given_value(
            self.data["GateV"][: self.up_down_idx], off_GateV
        )
        id_on: float = self.data["DrainI"][id_on_idx]
        id_off: float = self.data["DrainI"][id_off_idx]
        id_on_off: float = id_on / id_off
        return id_on, id_off, id_on_off

    def _calculate_threshold_voltage(
        self, slope_id_up: float, y_intercept_id_up: float
    ):
        """
        Function that calculates the threshold voltage of the OFET.

        Args:
            slope_id_up (float): slope of the up saturation regime
            y_intercept (float): y-intercept of the up saturation regime
        """
        # get Vthreshold by getting the x-intercept from the linear fit
        # y=mx+b, to get to x-intercept: y = 0, find x. -b/m = x
        threshold_voltage: float = -y_intercept_id_up / slope_id_up
        return threshold_voltage

    def _find_index_given_value(self, data: pd.Series, value: float):
        """
        Find the index in the column where the value is closest to the given value.
        """
        return data.sub(value).abs().idxmin()

    def _find_linear_regime_from_up_transfer_curve(
        self, threshold: float = 3, sampling_rate: int = 5
    ) -> tuple:
        """
        Find the linear regime of the up transfer curve.

        Returns:
            tuple: (slope, intercept, r2, max_x_range, min_x_range, linear_regime_start, linear_regime_end)
        """
        x = self.data["GateV"][0 : self.up_down_idx]
        y = self.data["sqrt_DrainI"][0 : self.up_down_idx]
        # calculate first normalized gradient
        first_derivative = np.gradient(y[::sampling_rate], x[::sampling_rate])
        first_norms = np.linalg.norm(first_derivative, axis=0)
        first_derivative = [
            np.where(first_norms == 0, 0, i / first_norms) for i in first_derivative
        ]
        # calculate second derivative
        second_derivative = np.gradient(first_derivative, x[::sampling_rate])
        second_norms = np.linalg.norm(second_derivative, axis=0)
        second_derivative = np.array(
            [
                np.where(second_norms == 0, 0, i / second_norms)
                for i in second_derivative
            ]
        )
        linear_regime_idx = np.split(
            linear_regime_idx, np.where(np.diff(linear_regime_idx) != 1)[0] + 1
        )[-1]
        print(linear_regime_idx)
        # get the first and last index of the linear regime
        linear_regime_start: int = linear_regime_idx[0] * sampling_rate
        linear_regime_end: int = linear_regime_idx[-1] * sampling_rate
        # calculate a linear fit for the linear regime
        slope, y_intercept = np.polyfit(
            x[linear_regime_start:linear_regime_end],
            y[linear_regime_start:linear_regime_end],
            1,
        )
        # calculate the correlation coefficient of the linear fit
        r2 = r2_score(
            x[linear_regime_start:linear_regime_end],
            np.polyval((slope, y_intercept), y[linear_regime_start:linear_regime_end]),
        )
        max_x_range = x[linear_regime_start]
        min_x_range = x[linear_regime_end]
        return (
            slope,
            y_intercept,
            r2,
            max_x_range,
            min_x_range,
            linear_regime_start,
            linear_regime_end,
        )

    def plot_transfer_curve(
        self,
        xlim: tuple = (-80, 0),
        max_x_range: float = -30,
        min_x_range: float = -70,
        auto_linear_regime: bool = False,
        direction: str = "up",
    ):
        """
        Plot the transfer curve of the OFET.

        Args:
            xlim (tuple[float]): x-axis limits for the plot
            max_x_range (float): maximum x range for the plot (i.e. -30)
            min_x_range (float): minimum x range for the plot (i.e. -70)
        """
        # Plot transfer curves
        fig, ax = plt.subplots(figsize=(3, 4))  # create a figure and axis object
        ax2 = ax.twinx()  # create a second y-axis that shares the same x-axis
        ax2.set_yscale("log")

        if direction == "up":
            # create mask for plotting within xlimits
            up_mask = (self.data["GateV"][0 : self.up_down_idx] >= xlim[0]) & (
                self.data["GateV"][0 : self.up_down_idx] <= xlim[1]
            )
            # Plot sqrt_DrainI for id_up
            ax.plot(
                self.data["GateV"][0 : self.up_down_idx][up_mask],
                self.data["sqrt_DrainI"][0 : self.up_down_idx][up_mask],
                color=self.up_color,
                label="id_up",
                linewidth=1.5,
                markersize=12,
                linestyle="dotted",
            )
            # plot linear line that depicts the linear regime for the up_transfer_curve
            # find index of the max and min x values for the up_transfer_curve
            if auto_linear_regime:
                (
                    slope,
                    y_intercept,
                    r2,
                    min_x_range,
                    max_x_range,
                    linear_regime_start,
                    linear_regime_end,
                ) = self._find_linear_regime_from_up_transfer_curve()
            else:
                linear_regime_start = self._find_index_given_value(
                    self.data["GateV"][0 : self.up_down_idx], min_x_range
                )
                linear_regime_end = self._find_index_given_value(
                    self.data["GateV"][0 : self.up_down_idx], max_x_range
                )
            self.max_x_range = max_x_range
            self.min_x_range = min_x_range
            ax.plot(
                [min_x_range, max_x_range],
                [
                    self.data["sqrt_DrainI"][linear_regime_start],
                    self.data["sqrt_DrainI"][linear_regime_end],
                ],
                color="red",
                label="linear_regime",
                linewidth=1.5,
                markersize=12,
                linestyle="--",
            )
            # plot DrainI and id_up
            ax2.plot(
                self.data["GateV"][0 : self.up_down_idx][up_mask],
                np.abs(self.data["DrainI"][0 : self.up_down_idx][up_mask]),
                color=self.up_color,
                label="$\mathregular{I_{D}}$, up",
                linewidth=1.5,
                markersize=12,
            )

        elif direction == "down":
            # create mask for plotting within xlimits
            down_mask = (self.data["GateV"][self.up_down_idx :] >= xlim[0]) & (
                self.data["GateV"][self.up_down_idx :] <= xlim[1]
            )
            # plot sqrt_DrainI for id_down
            ax.plot(
                self.data["GateV"][self.up_down_idx :][down_mask],
                self.data["sqrt_DrainI"][self.up_down_idx :][down_mask],
                color=self.down_color,
                label="id_down",
                linewidth=1.5,
                markersize=12,
                linestyle="dotted",
            )
            # plot DrainI and id_down
            ax2.plot(
                self.data["GateV"][self.up_down_idx :][down_mask],
                np.abs(self.data["DrainI"][self.up_down_idx :][down_mask]),
                color=self.down_color,
                label="$\mathregular{I_{D}}$, down",
                linewidth=1.5,
                markersize=12,
            )
        elif direction == "both":
            # create mask for plotting within xlimits
            up_mask = (self.data["GateV"][0 : self.up_down_idx] >= xlim[0]) & (
                self.data["GateV"][0 : self.up_down_idx] <= xlim[1]
            )
            # Plot sqrt_DrainI for id_up
            ax.plot(
                self.data["GateV"][0 : self.up_down_idx][up_mask],
                self.data["sqrt_DrainI"][0 : self.up_down_idx][up_mask],
                color=self.up_color,
                label="id_up",
                linewidth=1.5,
                markersize=12,
                linestyle="dotted",
            )
            # create mask for plotting within xlimits
            down_mask = (self.data["GateV"][self.up_down_idx :] >= xlim[0]) & (
                self.data["GateV"][self.up_down_idx :] <= xlim[1]
            )
            # plot sqrt_DrainI for id_down
            ax.plot(
                self.data["GateV"][self.up_down_idx :][down_mask],
                self.data["sqrt_DrainI"][self.up_down_idx :][down_mask],
                color=self.down_color,
                label="id_down",
                linewidth=1.5,
                markersize=12,
                linestyle="dotted",
            )
            # plot DrainI and id_up
            ax2.plot(
                self.data["GateV"][0 : self.up_down_idx][up_mask],
                np.abs(self.data["DrainI"][0 : self.up_down_idx][up_mask]),
                color=self.up_color,
                label="$\mathregular{I_{D}}$, up",
                linewidth=1.5,
                markersize=12,
            )
            # plot DrainI and id_down
            ax2.plot(
                self.data["GateV"][self.up_down_idx :][down_mask],
                np.abs(self.data["DrainI"][self.up_down_idx :][down_mask]),
                color=self.down_color,
                label="$\mathregular{I_{D}}$, down",
                linewidth=1.5,
                markersize=12,
            )
            # plot linear line that depicts the linear regime for the up_transfer_curve
            # find index of the max and min x values for the up_transfer_curve
            if auto_linear_regime:
                (
                    slope,
                    y_intercept,
                    r2,
                    min_x_range,
                    max_x_range,
                    linear_regime_start,
                    linear_regime_end,
                ) = self._find_linear_regime_from_up_transfer_curve()
            else:
                linear_regime_start = self._find_index_given_value(
                    self.data["GateV"][0 : self.up_down_idx], min_x_range
                )
                linear_regime_end = self._find_index_given_value(
                    self.data["GateV"][0 : self.up_down_idx], max_x_range
                )
            self.max_x_range = max_x_range
            self.min_x_range = min_x_range
            ax.plot(
                [min_x_range, max_x_range],
                [
                    self.data["sqrt_DrainI"][linear_regime_start],
                    self.data["sqrt_DrainI"][linear_regime_end],
                ],
                color="red",
                label="linear_regime",
                linewidth=1.5,
                markersize=12,
                linestyle="--",
            )

        else:
            raise ValueError(
                "Invalid direction. Please choose 'up', 'down', or 'both'."
            )
        ax.set_ylabel("$|\mathregular{I_{D}}|^{1/2}$ ($A^{1/2}$)")  # y-axis label
        ax.set_xlabel("V$_G$ (V)")  # x-axis label
        ax2.set_ylabel("$|\mathregular{I_{D}}|$ (A)")  # y-axis label

        ax2.legend(loc="upper right", frameon=False)  # legend for id_up and id_down

        # y-axis and x-axis ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(
            axis="both", which="minor", length=3, direction="in"
        )  # direction and length of minor ticks in X and Y-axis
        ax.tick_params(axis="both", which="major", length=4, direction="in")
        ax2.tick_params(
            axis="both", which="minor", direction="in"
        )  # direction and length of minor ticks in X and Y-axis
        ax2.tick_params(axis="both", which="major", length=4, direction="in")
        # Set scientific notation for ticks
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(xfmt)
        ax.yaxis.major.formatter._useMathText = True
        # Despines the figure
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        # TODO: Create arrows that point from the plot to the y-axis

        # Save the figure
        plt.savefig(
            self.result_dir / f"{self.root_name}/transfer_curve_{self.sheet_name}.svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.result_dir / f"{self.root_name}/transfer_curve_{self.sheet_name}.jpg",
            dpi=300,
            bbox_inches="tight",
        )

    def plot_up_saturation_regime(
        self, max_x_range: float = -30, min_x_range: float = -70
    ):
        """
        Plot the saturation regime of the OFET.

        Args:
            max_x_range (float): maximum x range for the plot (i.e. -30)
            min_x_range (float): minimum x range for the plot (i.e. -70)
        """
        fig, ax = plt.subplots()  # create a figure and axis object

        # find indicies of x values
        linear_regime_start = self._find_index_given_value(
            self.data["GateV"][0 : self.up_down_idx], max_x_range
        )
        linear_regime_end = self._find_index_given_value(
            self.data["GateV"][0 : self.up_down_idx], min_x_range
        )

        # plot sqrt_DrainI of saturation regime (up curve)
        ax.plot(
            self.data["GateV"][linear_regime_start:linear_regime_end],
            self.data["sqrt_DrainI"][linear_regime_start:linear_regime_end],
            color=self.up_color,
            label="sqrt_DrainI_up",
            linewidth=1.5,
            markersize=12,
        )  # plot sqrt_DrainI
        ax.set_ylabel("$|\mathregular{I_{D}}|^{1/2}$ ($A^{1/2}$)")  # y-axis label
        ax.set_xlabel("V$_G$ (V)")  # x-axis label

        # calculate linear fit
        slope_id_up, intercept, r_value, p_value, std_err = stats.linregress(
            self.data["GateV"][linear_regime_start:linear_regime_end],
            self.data["sqrt_DrainI"][linear_regime_start:linear_regime_end],
        )
        self.slope_id_up = slope_id_up
        self.y_intercept_id_up = intercept
        # get r2 of the linear fit
        r2: float = r2_score(
            self.data["sqrt_DrainI"][linear_regime_start:linear_regime_end],
            slope_id_up * self.data["GateV"][linear_regime_start:linear_regime_end]
            + intercept,
        )

        # plot linear fit
        ax.plot(
            self.data["GateV"][linear_regime_start:linear_regime_end],
            slope_id_up * self.data["GateV"][linear_regime_start:linear_regime_end]
            + intercept,
            color=self.up_color,
            label="linear_fit; slope: {:.7f}, intercept: {:.5f},\nR$^2$: {:.5f}".format(
                slope_id_up, intercept, r2
            ),
            linewidth=1.5,
            markersize=12,
            linestyle=":",
        )
        # Set scientific notation for ticks
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(xfmt)
        ax.yaxis.major.formatter._useMathText = True
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(
            axis="both", which="minor", length=3, direction="in"
        )  # direction and length of minor ticks in X and Y-axis
        ax.tick_params(axis="both", which="major", length=4, direction="in")

        # plot legend of slope, intercept, and R2
        ax.legend(loc="best", frameon=False)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.savefig(
            self.result_dir / f"{self.root_name}/up_saturation_{self.sheet_name}svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.result_dir / f"{self.root_name}/up_saturation_{self.sheet_name}.jpg",
            dpi=300,
            bbox_inches="tight",
        )

    def plot_down_saturation_regime(
        self, max_x_range: float = -30, min_x_range: float = -70
    ):
        """
        Plot the saturation regime of the OFET.

        Args:
            max_x_range (float): maximum x range for the plot (i.e. -30)
            min_x_range (float): minimum x range for the plot (i.e. -70)
        """
        fig, ax = plt.subplots()  # create a figure and axis object

        # find indicies of x values
        linear_regime_start = self._find_index_given_value(
            self.data["GateV"][self.up_down_idx :], min_x_range
        )
        linear_regime_end = self._find_index_given_value(
            self.data["GateV"][self.up_down_idx :], max_x_range
        )

        # plot sqrt_DrainI of saturation regime (down curve)
        ax.plot(
            self.data["GateV"][linear_regime_start:linear_regime_end],
            self.data["sqrt_DrainI"][linear_regime_start:linear_regime_end],
            color=self.down_color,
            label="sqrt_DrainI_down",
            linewidth=1.5,
            markersize=12,
        )  # plot sqrt_DrainI
        ax.set_ylabel("$|\mathregular{I_{D}}|^{1/2}$ ($A^{1/2}$)")  # y-axis label
        ax.set_xlabel("V$_G$ (V)")  # x-axis label

        # calculate the linear fit
        slope_id_down, intercept, r_value, p_value, std_err = stats.linregress(
            self.data["GateV"][linear_regime_start:linear_regime_end],
            self.data["sqrt_DrainI"][linear_regime_start:linear_regime_end],
        )
        self.slope_id_down = slope_id_down
        # get r2 of the linear fit
        r2 = r2_score(
            self.data["sqrt_DrainI"][linear_regime_start:linear_regime_end],
            slope_id_down * self.data["GateV"][linear_regime_start:linear_regime_end]
            + intercept,
        )
        # plot linear fit
        ax.plot(
            self.data["GateV"][linear_regime_start:linear_regime_end],
            slope_id_down * self.data["GateV"][linear_regime_start:linear_regime_end]
            + intercept,
            color=self.down_color,
            label="linear_fit; slope: {:.7f}, intercept: {:.5f}, \nR$^2$: {:.5f}".format(
                slope_id_down, intercept, r2
            ),
            linewidth=1.5,
            markersize=12,
            linestyle=":",
        )
        # Set scientific notation for ticks
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(xfmt)
        ax.yaxis.major.formatter._useMathText = True
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(
            axis="both", which="minor", length=3, direction="in"
        )  # direction and length of minor ticks in X and Y-axis
        ax.tick_params(axis="both", which="major", length=4, direction="in")

        # plot legend of slope, intercept, and R2
        ax.legend(loc="best", frameon=False)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.savefig(
            self.result_dir / f"{self.root_name}/down_saturation_{self.sheet_name}.svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.result_dir / f"{self.root_name}/down_saturation_{self.sheet_name}.jpg",
            dpi=300,
            bbox_inches="tight",
        )

    def plot_id_vg(self):
        """
        Plot the DrainI vs GateV curve.
        """
        pass

    def export_metadata(self, on_GateV: float = -30, off_GateV: float = -70):
        """
        Export the metadata of the OFET.
        NOTE: only use this once you've plotted everything (transfer curves and saturation regimes)
        Metadata (avg and stdev): mobility, on/off ratio, threshold voltage
        """
        # calculate mobility
        u_down, u_up, u_average = self._calculate_mobility(
            self.slope_id_up, self.slope_id_down
        )
        # calculate id_on_off_ratio
        id_on, id_off, id_on_off = self._calculate_id_on_off_ratio(on_GateV, off_GateV)
        # calculate threshold_voltage
        threshold_voltage = self._calculate_threshold_voltage(
            self.slope_id_up, self.y_intercept_id_up
        )
        metadata: pd.DataFrame = pd.DataFrame.from_dict(
            {
                self.sheet_name: [
                    self.sheet_name,
                    u_up,
                    u_down,
                    u_average,
                    id_on,
                    id_off,
                    id_on_off,
                    threshold_voltage,
                ]
            },
            orient="index",
            columns=[
                "Sheet Name",
                "Up Mobility (cm^2/Vs)",
                "Down Mobility (cm^2/Vs)",
                "Avg (of Up and Down) Mobility (cm^2/Vs)",
                "ID_on (A)",
                "ID_off (A)",
                "ID_on_off_ratio",
                "Threshold Voltage (V)",
            ],
        )
        # add to .csv if present
        if os.path.isfile(
            self.result_dir / f"{self.root_name}/metadata_{self.root_name}.csv"
        ):
            metadata_old = pd.read_csv(
                self.result_dir / f"{self.root_name}/metadata_{self.root_name}.csv"
            )
            # drop the row if the sheet name is already present
            if metadata_old["Sheet Name"].isin([self.sheet_name]).any():
                metadata_old = metadata_old.drop(
                    metadata_old[metadata_old["Sheet Name"] == self.sheet_name].index
                )
            metadata = pd.concat([metadata_old.iloc[:-2], metadata])

        # replace bottom line with new average and stdev
        # calculate average and stdev of mobility, id_on_off_ratio, threshold_voltage
        # average each column individually
        u_down_across_replicates: float = metadata["Down Mobility (cm^2/Vs)"].mean()
        u_up_across_replicates: float = metadata["Up Mobility (cm^2/Vs)"].mean()
        u_average_across_replicates: float = metadata[
            "Avg (of Up and Down) Mobility (cm^2/Vs)"
        ].mean()
        id_on_average_across_replicates: float = metadata["ID_on (A)"].mean()
        id_off_average_across_replicates: float = metadata["ID_off (A)"].mean()
        id_on_off_average_across_replicates: float = metadata["ID_on_off_ratio"].mean()
        vthreshold_average: float = metadata["Threshold Voltage (V)"].mean()

        # standard deviation of each column
        u_down_std: float = metadata["Down Mobility (cm^2/Vs)"].std()
        u_up_std: float = metadata["Up Mobility (cm^2/Vs)"].std()
        n: int = len(metadata)
        u_average_std: float = np.sqrt(
            (n * u_down_std**2 + n * u_up_std**2) / (n + n)
        )  # https://www.geeksforgeeks.org/combined-standard-deviation-meaning-formula-and-example/
        # add a row to the dataframe with the average mobility
        id_on_std: float = metadata["ID_on (A)"].std()
        id_off_std: float = metadata["ID_off (A)"].std()
        id_on_off_std: float = metadata["ID_on_off_ratio"].std()
        vthreshold_std: float = metadata["Threshold Voltage (V)"].std()
        metadata.loc["Average"] = [
            "Average",
            u_up_across_replicates,
            u_down_across_replicates,
            u_average_across_replicates,
            id_on_average_across_replicates,
            id_off_average_across_replicates,
            id_on_off_average_across_replicates,
            vthreshold_average,
        ]
        metadata.loc["Std"] = [
            "Std",
            u_down_std,
            u_up_std,
            u_average_std,
            id_on_std,
            id_off_std,
            id_on_off_std,
            vthreshold_std,
        ]

        metadata.to_csv(
            self.result_dir / f"{self.root_name}/metadata_{self.root_name}.csv",
            mode="w",
            index=False,
        )


# TODO:
class OutputCurve:
    """
    Class that contains all the methods to plot the output curve.
    """

    def __init__(
        self,
        data_path: Path,
        sheet_name: str,
        result_dir: Path,
        style_path: Path,
    ):
        """Initialize class with data directory, result directory.

        Args:
            data_dir (Path): path file to .csv with data
            result_dir (Path): result directory to save plots
            c (float): _description_ (F/cm^2)
            w (float): channel width (cm)
            l (float): channel length (cm)
        """
        self.data_dir = data_path
        self.sheet_name = sheet_name
        self.root_name = self.data_dir.stem
        self.result_dir = result_dir
        with open(style_path, "r") as f:  # opens the style.json file
            self.style: dict = json.load(f)  # loads as a dictionary object
        self.data = pd.read_excel(self.data_dir, self.sheet_name)
        # if directory is not present, make the directory
        if not os.path.isdir(self.result_dir / self.root_name):
            os.makedirs(self.result_dir / self.root_name)

    def _find_index_given_value(self, data: pd.Series, value: float):
        """
        Find the index in the column where the value is closest to the given value.
        """
        return data.sub(value).abs().idxmin()

    def plot_output_curve(
        self,
        annotate_x: float = -30,
        num_of_annotate: int = 5,
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
            "green_blue_10",
        ],
    ):
        """
        Plot the output curve of the OFET.

        Args:
            annotate_x (float): x value to align the annotations
            num_of_annotate (int): number of plots to annotate (starting from the last one)
        """
        # Find all the columns with "DrainV("", and then plot them all
        num_of_output_curves = 1
        for column in self.data.columns:
            if "DrainV(" in column:
                num_of_output_curves += 1
        # Check if there are enough colors
        assert (
            len(color_order) >= num_of_output_curves
        ), f"{len(color_order)=}, {num_of_output_curves=}, Number of colors in color_order must match or be greater than the number of columns in the data file."
        fig, ax = plt.subplots(figsize=(6, 4))
        color_idx = 0
        for num in range(1, num_of_output_curves):
            gate_voltage_column_name: str = f"GateV({num})"
            ax.plot(
                self.data[f"DrainV({num})"],
                -self.data[f"DrainI({num})"],
                label=f"$V_G$={self.data[gate_voltage_column_name][0]}",
                linewidth=1.5,
                markersize=12,
                color=self.style["color"][color_order[color_idx]],
            )  # plot sqrt_DrainI
            color_idx += 1
            # annotate each plot with the correct text label
            # NOTE: change how many you want to plot (ex. last 6)
            # find the index of the defined value
            if color_idx >= (num_of_output_curves - num_of_annotate):
                idx = self._find_index_given_value(
                    self.data[f"DrainV({num})"], annotate_x
                )
                ax.annotate(
                    f"$V_G$={self.data[gate_voltage_column_name][0]}",
                    (
                        self.data[f"DrainV({num})"].iloc[idx],
                        -self.data[f"DrainI({num})"].iloc[idx],
                    ),
                    textcoords="offset points",
                    xytext=(5, 4),
                    ha="center",
                )
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles[::-1], labels[::-1], loc="best", frameon=False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_ylabel("-I$_D$ (A)")  # y-axis label
        ax.set_xlabel("V$_D$ (V)")  # x-axis label
        # y-axis and x-axis ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(
            axis="both", which="minor", length=4, direction="in"
        )  # direction and length of minor ticks in X and Y-axis
        ax.tick_params(axis="both", which="major", length=6, direction="in")

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
            self.result_dir / f"{self.root_name}/output_curve_{self.root_name}.svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.result_dir / f"{self.root_name}/output_curve_{self.root_name}.jpg",
            dpi=300,
            bbox_inches="tight",
        )


# TODO:
class OverlayTransferCurves:
    """
    Class that contains all the methods to overlay transfer curves (without saturation regime).
    """

    def __init__(
        self,
        data_paths: dict,
        data_dir: Path,
        result_dir: Path,
        overlay_transfer_curve_name: str,
        style_path: Path,
    ):
        self.data_dicts = data_paths
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.overlay_transfer_curve_name = overlay_transfer_curve_name
        self.root_name = self.data_dir.stem
        with open(style_path, "r") as f:
            self.style: dict = json.load(f)

    def _find_index_given_value(self, data: pd.Series, value: float):
        """
        Find the index in the column where the value is closest to the given value.
        """
        return data.sub(value).abs().idxmin()

    def _find_up_down_idx(self, data: pd.DataFrame):
        """
        Segment the data into up curve and down curve
        Look for index where in GateV column, there are two consecutives values that are equivalent
        This is the index where the up curve ends and the down curve starts
        """
        up_down_idx: int = data["GateV"].diff().eq(0).idxmax()
        return up_down_idx

    def plot_overlay_transfer_curve(
        self, labels: list = [], color_order: list = ["blue", "yellow"]
    ):
        """
        It is very similar to the plot_transfer_curve method from TransferCurve class.
        However, loads data differently to plot several transfer curves on the same plot.
        """
        self.color_order = color_order
        assert len(self.color_order) == len(
            self.data_dicts
        ), "Number of colors in color_order must match the number of data files in data_paths."
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_yscale("log")
        color_idx = 0
        for data_path, sheet_name in self.data_dicts.items():
            data_path = Path(self.data_dir / data_path)
            data = pd.read_excel(data_path, sheet_name=sheet_name)
            up_down_idx = self._find_up_down_idx(data)
            if labels == None:
                plot_label = sheet_name
            else:
                plot_label = labels[color_idx]
            ax.plot(
                data["GateV"][:up_down_idx],
                np.abs(data["DrainI"][:up_down_idx]),
                label=plot_label,
                linewidth=1.5,
                markersize=12,
                color=self.style["color"][self.color_order[color_idx]],
            )
            color_idx += 1
        ax.legend(loc="best", frameon=False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_ylabel("|I$_D$| (A)")
        ax.set_xlabel("V$_G$ (V)")
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=3, direction="in")
        ax.tick_params(axis="both", which="major", length=4, direction="in")
        plt.savefig(
            self.result_dir
            / f"overlay_transfer_curve_{self.overlay_transfer_curve_name}.svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.result_dir
            / f"overlay_transfer_curve_{self.overlay_transfer_curve_name}.jpg",
            dpi=300,
            bbox_inches="tight",
        )


class MW_Curves:
    """
    Class that contains all the methods to plot mobility, modulus, COS, strain vs molecular weight curve.
    """

    def __init__(self, data_path: Path, result_dir: Path, style_path: Path):
        self.data_path = data_path
        self.result_dir = result_dir
        with open(style_path, "r") as f:  # opens the style.json file
            self.style: dict = json.load(f)

    def plot_mobility_vs_mw(self, color: str = "blue", name: str = "IDTBT"):
        """
        Plot avg_mobility (with stdev) vs molecular weight curve.
        Args:
            color (str): color of the plot
            name (str): name of the plot
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        data = pd.read_excel(self.data_path)
        ax.errorbar(
            data["Molecular Weight"],
            data["Mobility_avg"],
            yerr=data["Mobility_stdev"],
            fmt="o",
            color=self.style["color"][color],
        )
        ax.set_ylabel("Mobility (cm$^2$/Vs)")
        ax.set_xlabel("Molecular Weight (g/mol)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=4, direction="in")
        ax.tick_params(axis="both", which="major", length=6, direction="in")
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(xfmt)
        ax.yaxis.major.formatter._useMathText = True
        plt.savefig(
            self.result_dir / f"{name}_mobility_vs_mw.svg", dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            self.result_dir / f"{name}_mobility_vs_mw.jpg", dpi=300, bbox_inches="tight"
        )

    def plot_modulus_cos_strain_vs_mw(
        self,
        modulus_color: str = "blue",
        cos_color: str = "green",
        strain_color: str = "yellow",
        name: str = "IDTBT",
    ):
        """
        Plot modulus, COS, and strain vs molecular weight curve.
        Args:
            modulus_color (str): color of the modulus plot
            cos_color (str): color of the COS plot
            strain_color (str): color of the strain plot
            name (str): name of the plot
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        data = pd.read_excel(self.data_path)
        # Plot modulus vs. mw
        ax.errorbar(
            data["Molecular Weight"],
            data["Modulus"],
            yerr=data["Modulus_stdev"],
            fmt="o",
            color=self.style["color"][modulus_color],
        )
        ax.set_ylabel("Modulus")
        ax.set_xlabel("Molecular Weight (g/mol)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=4, direction="in")
        ax.tick_params(axis="both", which="major", length=6, direction="in")
        xfmt = ScalarFormatter()
        xfmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(xfmt)
        ax.yaxis.major.formatter._useMathText = True
        # Plot COS vs. mw
        ax2 = ax.twinx()
        ax2.errorbar(
            data["Molecular Weight"],
            data["COS"],
            yerr=data["COS_stdev"],
            fmt="o",
            color=self.style["color"][cos_color],
        )
        ax2.set_ylabel("COS")
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.tick_params(axis="both", which="minor", length=4, direction="in")
        ax2.tick_params(axis="both", which="major", length=6, direction="in")
        ax2.yaxis.set_major_formatter(xfmt)
        ax2.yaxis.major.formatter._useMathText = True
        # Plot Strain vs. mw
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))

        ax3.errorbar(
            data["Molecular Weight"],
            data["Strain"],
            yerr=data["Strain_stdev"],
            fmt="o",
            color=self.style["color"][strain_color],
        )
        ax3.set_ylabel("Strain")
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax3.tick_params(axis="both", which="minor", length=4, direction="in")
        ax3.tick_params(axis="both", which="major", length=6, direction="in")
        ax3.yaxis.set_major_formatter(xfmt)
        ax3.yaxis.major.formatter._useMathText = True

        plt.savefig(
            self.result_dir / f"{name}_modulus_cos_strain_vs_mw.svg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            self.result_dir / f"{name}_modulus_cos_strain_vs_mw.jpg",
            dpi=300,
            bbox_inches="tight",
        )
