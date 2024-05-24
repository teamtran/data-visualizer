import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pdb
from sklearn.metrics import r2_score
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.ticker import ScalarFormatter
import json
import os

# TODO: add .svg plot and put legend outside of plot


class MechanicalPlot:
    def __init__(
        self,
        data_path: Path,
        result_path: Path,
        style_path: Path,
        cycle_label: str = "1-Stretch",  # "1-Compress"
        test_type: str = "tensile",
    ):
        """
        Args:
            data_path (Path): path to the data file
            result_path (Path): path to the result directory
            style_path (Path): path to the style file
        """
        self.data_path = data_path
        self.result_path = result_path
        self.style_path = style_path
        # load style .json as dictionary
        self.style = json.load(open(style_path))
        self.data = pd.read_csv(data_path)
        self.cycle_label = cycle_label
        self.test_type = test_type
        self.root_name = data_path.stem
        # if directory is not present, make the directory
        if not os.path.isdir(self.result_path):
            os.makedirs(self.result_path)

    def _preprocess_data(self):
        # Preprocess data (calculate strain and stress)
        # Find row index when Cycle is 1-Stretch
        self.start_idx = self.data.index[self.data["Cycle"] == self.cycle_label][0]
        self.initial_size = self.data["Size_mm"][self.start_idx]

        # Calculating Strain
        self.data["Strain (%)"] = self.data["Displacement_mm"] * 100 / self.initial_size
        self.data["Stress (MPa)"] = (
            self.data["Force_N"] / self.data["cross sectional area"][0]
        )
        # Find row index when stress is near 0
        # error in stress calculation
        self.stress_error = (
            self.data["load cell"][0] * 0.02 / self.data["cross sectional area"][0]
        )
        if self.test_type == "tensile":
            # find the row index at which the stress is maximum
            self.stress_maximum = self.data["Stress (MPa)"].max()
            self.stress_maximum_index = self.data.index[
                self.data["Stress (MPa)"] == self.stress_maximum
            ][0]
            self.end_idx = self.data[self.stress_maximum_index :].index[
                self.data["Stress (MPa)"][self.stress_maximum_index :]
                < self.stress_error
            ][0]
        elif self.test_type == "compression":
            # find the row index at which the stress hits a local maximum
            # Smooth the data
            self.data["Smoothed Stress (MPa)"] = (
                self.data["Stress (MPa)"].rolling(5).mean()
            )
            # calculate the first derivative of the stress
            first_derivative = pd.Series(
                np.gradient(self.data["Smoothed Stress (MPa)"], self.data["Strain (%)"])
            )
            # find the row index at which the first derivative goes from positive to negative
            self.stress_maximum_index = first_derivative.index[first_derivative < 0][0]
            print(self.stress_maximum_index)
            # find the row index at which the stress is near 0
            self.end_idx = self.data.index[self.data["Cycle"] == self.cycle_label][-1]
        else:
            raise ValueError("type should be either 'tensile' or 'compression'")

    def _find_linear_regime(
        self, threshold: float = 0.01, sampling_rate: int = 5
    ) -> tuple:
        """
        Find the linear regime of the up transfer curve.

        Args:
            threshold (float, optional): The threshold for the second derivative to be considered 0. Defaults to 0.01. (should fall within 0 to 1)
            sampling_rate (int, optional): The sampling rate for the second derivative. Defaults to 5.

        Returns:
            tuple: (linear_fit, r2, linear_regime_start, linear_regime_end)
        """
        if self.test_type == "compression":
            linear_regime_choice: int = -1  # last linear regime
            x = self.data["Strain (%)"][: self.stress_maximum_index]
            y = self.data["Stress (MPa)"][: self.stress_maximum_index]
            print(x, y)
        else:
            linear_regime_choice: int = 0  # first linear regime
            x = self.data["Strain (%)"]
            y = self.data["Stress (MPa)"]
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
        # find the row indices at which the second derivative is 0 +/- threshold
        linear_regime_idx = np.where(
            (second_derivative < (threshold)) & (second_derivative > -(threshold))
        )[0]
        # get index at which the linear_regime_idx does not have consecutive values
        # because non-consecutive value means a gap between linear segments of the curve
        print(linear_regime_idx)
        linear_regime_idx = np.split(
            linear_regime_idx, np.where(np.diff(linear_regime_idx) != 1)[0] + 1
        )[linear_regime_choice]
        print(linear_regime_idx)
        # get the first and last index of the linear regime
        linear_regime_start: int = linear_regime_idx[0] * sampling_rate
        linear_regime_end: int = linear_regime_idx[-1] * sampling_rate
        # calculate a linear fit for the linear regime
        linear_fit = np.polyfit(
            x[linear_regime_start:linear_regime_end],
            y[linear_regime_start:linear_regime_end],
            1,
        )
        # calculate the correlation coefficient of the linear fit
        r2 = r2_score(
            y[linear_regime_start:linear_regime_end],
            np.polyval(linear_fit, x[linear_regime_start:linear_regime_end]),
        )
        return (
            linear_fit,
            r2,
            linear_regime_start,
            linear_regime_end,
        )

    def plot_curve(
        self,
        color_scatter: str = "dark_blue",
        color_linear_fit: str = "red",
        linear_regime_threshold: float = 0.01,
        linear_regime_sampling_rate: float = 5,
    ):
        # Plot data
        fig, ax = plt.subplots()
        ax.scatter(
            self.data["Strain (%)"][self.start_idx : self.end_idx + 5],
            self.data["Stress (MPa)"][self.start_idx : self.end_idx + 5],
            label="Stress-Strain Curve",
            color=self.style["color"][color_scatter],
            marker="o",
            s=4,
        )
        # Calculate linear fit automatically
        linear_fit, r2, linear_regime_start, linear_regime_end = (
            self._find_linear_regime(
                threshold=linear_regime_threshold,
                sampling_rate=linear_regime_sampling_rate,
            )
            # self.data.loc[self.start_idx + 1 : self.stress_maximum_index]
        )
        # plot linear fit and statistics
        ax.plot(
            self.data["Strain (%)"][linear_regime_start:linear_regime_end],
            np.polyval(
                linear_fit,
                self.data["Strain (%)"][linear_regime_start:linear_regime_end],
            ),
            label=f"Linear Fit: y = {linear_fit[0]:.4f}x + {linear_fit[1]:.4f} \n R^2 = {r2:.4f}",
            color=self.style["color"][color_linear_fit],
        )

        ax.set_xlabel("Strain (%)")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title(f"{self.root_name}")
        # Plot legend outside of the plot to the right
        ax.legend(loc="center right", bbox_to_anchor=(1.7, 0.5))
        # y-axis and x-axis ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(
            axis="both", which="minor", length=4, direction="in"
        )  # direction and length of minor ticks in X and Y-axis
        ax.tick_params(axis="both", which="major", length=6, direction="in")
        # Set scientific notation for ticks
        # xfmt = ScalarFormatter()
        # xfmt.set_powerlimits((-3, 3))
        # ax.yaxis.set_major_formatter(xfmt)
        # ax.yaxis.major.formatter._useMathText = True
        # Despines the figure
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Add text below the plot with the linear fit threshold and sampling rate
        ax.text(
            0.1,
            -0.2,
            f"Linear Regime Threshold: {linear_regime_threshold} \n Linear Regime Sampling Rate: {linear_regime_sampling_rate}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        # Save the plot
        plt.savefig(
            self.result_path / f"{self.root_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            self.result_path / f"{self.root_name}.svg", dpi=300, bbox_inches="tight"
        )

    def plot_cycle_curves(
        self,
        setname_order: list[str] = [
            "Tension1 petg rate cycle 0.01",
            "Tension1 petg rate cycle 0.1",
            "Tension1 petg rate cycle 0.2",
            "Tension1 petg rate cycle 2",
            "Tension1 petg rate cycle 5",
        ],
        color_order: list[str] = [
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
        fig, ax = plt.subplots(figsize=(6, 4))
        for setname in setname_order:
            setname_start_index = self.data.loc[self.data["SetName"] == setname].index[
                0
            ]
            setname_end_index = self.data.loc[self.data["SetName"] == setname].index[-1]
            # Plot data
            ax.scatter(
                self.data["Strain (%)"][setname_start_index:setname_end_index],
                self.data["Stress (MPa)"][setname_start_index:setname_end_index],
                label=setname,
                color=self.style["color"][color_order[setname_order.index(setname)]],
                s=5,
            )
        # Plot legend outside of the plot to the right
        ax.legend(loc="center right", bbox_to_anchor=(1.7, 0.5))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("Strain (%)")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title(f"{self.root_name}")
        # y-axis and x-axis ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(
            axis="both", which="minor", length=4, direction="in"
        )  # direction and length of minor ticks in X and Y-axis
        ax.tick_params(axis="both", which="major", length=6, direction="in")

        plt.savefig(
            self.result_path / f"{self.root_name}.png", dpi=300, bbox_inches="tight"
        )
