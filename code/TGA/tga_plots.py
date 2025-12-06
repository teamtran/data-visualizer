from typing import Optional
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
os.environ["MKL_THREADING_LAYER"] = "GNU"
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import scipy.stats as stats
from scipy.stats import linregress
from scipy.optimize import curve_fit
import pandas as pd
import json
import os
from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from sklearn.metrics import r2_score
from matplotlib.ticker import ScalarFormatter
import pdb


def _find_empty_row(data: pd.DataFrame):
    # get length of dataframe to get index of empty row
    num_skiprows = len(data) + 3

    return num_skiprows


class TGAPlots:
    """
    Class that contains methods to plot TGA-MS (dynamic, isothermal) data.
    """

    def __init__(
        self,
        data_dir: Path,
        tga_data_path: list[Path],
        ms_data_path: list[Path],
        labels: list[str],
        colors: list[str],
        result_dir: Path,
        style_path: Path,
    ):
        """
        Initialize the class with the data path and the result directory.
        """
        self.data_dir = data_dir
        self.tga_data_path = tga_data_path
        self.ms_data_path = ms_data_path
        self.labels = labels
        self.colors = colors
        self.result_dir = result_dir
        self.style_path = style_path
        self.style = json.load(open(style_path))
        self.result_name = ""
        for label in labels:
            self.result_name += label + "_"

    def preprocess(
        self,
        tga_data: pd.DataFrame,
        ms_data: pd.DataFrame,
        initial_correction: float,
        time_or_temp: str = "Time",
    ) -> pd.DataFrame:
        """Function that applies transformation to the dataframe which will make it ready for plotting. Note, this is specific to TGA-MS."""
        # Process TGA data
        # Truncate Temp./C column to Temp
        new_columns = tga_data.columns.tolist()
        new_columns[0] = "Temp"
        for column in new_columns:
            if "Mass" in column:
                new_columns[new_columns.index(column)] = "Mass loss/pct"
        tga_data.columns = new_columns

        # Process MS data with try-except
        try:
            # Truncate Temp./C column to Temp in ms_raw_data
            ms_data_columns = ms_data.columns.tolist()
            ms_data_columns[0] = "Temp"
            ms_data.columns = ms_data_columns

            # Find the row closest to the initial_correction for ms_data
            if time_or_temp == "Time":
                initial_correction_row_ms: int = ms_data.iloc[
                    (ms_data["Time/min"] - initial_correction).abs().argsort()[:1]
                ].index[0]
            else:
                initial_correction_row_ms: int = ms_data.iloc[
                    (ms_data["Temp"] - initial_correction).abs().argsort()[:1]
                ].index[0]

            # Remove the rows before the initial_correction_time_row for ms_data
            ms_data = ms_data.iloc[initial_correction_row_ms:]

        except Exception as e:
            # Set ms_data to None or empty DataFrame if processing fails
            ms_data = pd.DataFrame()

        # Account for uncertainty: balance drift (0.002mg/hr); balance uncertainty (2.5e-5mg)
        # tga_data["mass_loss_uncertainty"] = (
        #     tga_data["Time/min"] * (0.002 / 60) + 0.000025
        # )

        # tga_data["mass_loss_pct_uncertainty"] = (
        #     tga_data["mass_loss_uncertainty"] * 100 / initial_mass
        # )

        # Find the row closest to the initial_correction_time for tga_data
        if time_or_temp == "Time":
            initial_correction_row: int = tga_data.iloc[
                (tga_data["Time/min"] - initial_correction).abs().argsort()[:1]
            ].index[0]
        else:
            initial_correction_row: int = tga_data.iloc[
                (tga_data["Temp"] - initial_correction).abs().argsort()[:1]
            ].index[0]

        # Subtract 0 from the Mass loss/mg datapoint from the initial_correction_time_row
        correction_mass = 100 - tga_data["Mass loss/pct"][initial_correction_row]
        tga_data["Mass loss/pct"] = tga_data["Mass loss/pct"] + correction_mass

        # Remove the rows before the initial_correction_time_row for tga_data
        tga_data = tga_data.iloc[initial_correction_row:]

        return tga_data, ms_data

    def get_mass_at_time(self, time: float, tga_data: pd.DataFrame) -> float:
        """
        Get the mass at a specific time for the TGA data.
        """
        mass_pct = (
            tga_data["Mass loss/pct"]
            .iloc[(tga_data["Time/min"] - time).abs().argsort()[:1]]
            .values[0]
        )

        return mass_pct

    def plot_tga_isothermal(
        self,
        isothermal_temp: float,
        target_mass: int = 104,
        xlim: tuple = (0, 1450),
        ylim: tuple = (0, 100),
        initial_correction_time: int = 50,
        uncertainty: bool = False,
        time_for_mass_difference: float = 1450,
    ):
        """
        Plot several TGA isothermal data for comparison (can handle 1 or more).
        """
        fig, ax = plt.subplots(1, figsize=(6, 4))
        plt.subplots_adjust(hspace=0.5)
        # aesthetics
        ax.set_xlabel("Time (min)", fontsize=10)
        ax.set_ylabel("Mass (%)", fontsize=10)
        # ax.set_title(f"Isothermal TGA at {isothermal_temp}°C", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8, direction="in")
        # set xlim for the plots
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # ax[1].set_ylim(0, 0.2e-12)
        mass_difference_at_time = []
        for tga_path, ms_path, label, color in zip(
            self.tga_data_path, self.ms_data_path, self.labels, self.colors
        ):
            tga_data = pd.read_csv(
                self.data_dir / tga_path, encoding="iso-8859-1", on_bad_lines="skip"
            )
            initial_mass: float = float(tga_data.at[17, tga_data.columns[1]])
            num_skiprows: int = _find_empty_row(tga_data)
            tga_data = pd.read_csv(
                self.data_dir / tga_path,
                encoding="iso-8859-1",
                on_bad_lines="skip",
                skiprows=num_skiprows,
            )
            tga_data, ms_data = self.preprocess(
                tga_data, None, initial_correction_time, "Time"
            )
            mass_difference_at_time.append(
                100 - self.get_mass_at_time(time_for_mass_difference, tga_data)
            )
            ax.plot(
                tga_data["Time/min"] - initial_correction_time,
                tga_data["Mass loss/pct"],
                label=label,
                color=color,
                linewidth=1,
            )
            if uncertainty:
                ax.fill_between(
                    tga_data["Time/min"],
                    tga_data["Mass loss/pct"] - tga_data["mass_loss_pct_uncertainty"],
                    tga_data["Mass loss/pct"] + tga_data["mass_loss_pct_uncertainty"],
                    alpha=0.3,
                    facecolor=color,
                )
        ax.axhline(
            y=100,
            color="r",
            linestyle="--",
            linewidth=0.5,
            label="100% Mass",
        )
        ax.set_yticks([0, 20, 40, 60, 80, 100])  # Add this line
        # ax.grid(True, linestyle="-", alpha=0.2, linewidth=0.5, color="gray")
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(
            self.result_dir
            / f"{self.result_name}tga_isothermal_{isothermal_temp}_{target_mass}m_z.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir
            / f"{self.result_name}tga_isothermal_{isothermal_temp}_{target_mass}m_z.eps",
            format="eps",
            dpi=400,
        )
        print(
            f"Mass difference at {time_for_mass_difference} min: {mass_difference_at_time}"
        )

    def plot_tga_dynamic(
        self,
        target_mass: int = 104,
        t_depolymerization_cutoff: float = 99.5,
        xlim: tuple = (100, 400),
        ylim: tuple = (-5, 105),
        initial_correction_temp: int = 230,
    ):
        # TODO: initial correction temp instead of time
        """
        Plot the TGA dynamic data.
        """
        fig, ax = plt.subplots(1, figsize=(4, 3.25))
        plt.subplots_adjust(hspace=0.5)
        # aesthetics
        ax.set_xlabel("Temp ($^{o}$C)", fontsize=10)
        ax.set_ylabel("Mass (%)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8, direction="in")
        for tga_path, ms_path, label, color in zip(
            self.tga_data_path, self.ms_data_path, self.labels, self.colors
        ):
            tga_data = pd.read_csv(
                self.data_dir / tga_path,
                encoding="iso-8859-1",
                on_bad_lines="skip",
                skiprows=37,
            )
            ms_data = pd.read_csv(
                self.data_dir / ms_path,
                encoding="iso-8859-1",
                on_bad_lines="skip",
                skiprows=29,
            )
            tga_data, ms_data = self.preprocess(
                tga_data, None, initial_correction_temp, "Temp"
            )
            # find the index at which the mass loss is closest to the t_depolymerization_cutoff
            t_depolymerization_temp = (
                tga_data["Temp"]
                .iloc[
                    (tga_data["Mass loss/pct"] - t_depolymerization_cutoff)
                    .abs()
                    .argsort()[:1]
                ]
                .values[0]
            )
            ax.plot(
                tga_data["Temp"], tga_data["Mass loss/pct"], label=label, color=color
            )
            # Set colors for the axis labels
            # Draw lines for onset temperature determination
            (
                baseline_slope,
                baseline_intercept,
                gradient_slope,
                gradient_intercept,
                onset_temp,
            ) = self.calculate_onset_t(tga_data, xlim=xlim)
            # Create temperature range for plotting the extrapolated lines
            temp_range = np.linspace(xlim[0], xlim[1], 100)

            # Plot baseline extrapolation line
            baseline_line = baseline_slope * temp_range + baseline_intercept
            ax.plot(
                temp_range,
                baseline_line,
                ":",
                linewidth=1,
                color=color,
                alpha=0.8,
            )

            # Plot gradient extrapolation line
            gradient_line = gradient_slope * temp_range + gradient_intercept
            ax.plot(
                temp_range,
                gradient_line,
                ":",
                linewidth=1,
                color=color,
                alpha=0.8,
            )

            # Mark onset temperature point
            onset_mass_loss = baseline_slope * onset_temp + baseline_intercept
            ax.plot(
                onset_temp,
                onset_mass_loss,
                "x",
                markersize=6,
                label=label
                + "\n"
                + f"(T{'$_{onset}$'} = {onset_temp:.1f}°C), "
                + f"(T{'$_{depoly}$'} = {t_depolymerization_temp:.1f}°C)",
                color=color,
                zorder=5,
            )
            print(
                f"Name: {self.labels}, Onset Temperatures: {onset_temp}, Depolymerization Temperatures: {t_depolymerization_temp}"
            )

        # Draw a line in the y-axis at y=99%
        ax.axhline(
            y=t_depolymerization_cutoff,
            color="r",
            linestyle="--",
            linewidth=0.3,
            label="$T_\mathregular{depoly.}$"
            + f" at {t_depolymerization_cutoff}% Mass",
        )

        ax.legend(fontsize=6)
        ax.set_xlim(xlim)
        ax.set_yticks([0, 20, 40, 60, 80, 100])  # Add this line
        ax.set_ylim(ylim)
        plt.tight_layout()

        # ax.grid(True, linestyle="-", alpha=0.2, linewidth=0.5, color="gray")
        plt.savefig(
            self.result_dir / f"{self.result_name}tga_dynamic_{target_mass}m_z.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir / f"{self.result_name}tga_dynamic_{target_mass}m_z.eps",
            format="eps",
            dpi=400,
        )

    def plot_ms_peak_area(
        self,
        tga_type: str,
        isothermal_temp: Optional[float],
        initial_time: float = 40,
        end_time: float = 1450,
        m_z_start: int = 60,
        m_z_end: int = 150,
        normalize: bool = False,
        alpha_list: list[float] = [0.8, 0.5],
    ):
        """
        Plot the peak area for each m/z over time.
        x-axis: m/z
        y-axis: peak area from time 0 to time x.
        """
        fig, ax = plt.subplots(1, figsize=(4, 3.25))
        # aesthetics
        ax.set_xlabel("m/z", fontsize=5)
        ax.set_ylabel("Normalized Peak Area of Ion Current over Time (A)", fontsize=5)
        # if tga_type == "isothermal":
        #     ax.set_title(
        #         f"Normalized Peak Area for m/z {m_z_start} to {m_z_end} for {tga_type} TGA-MS data at {isothermal_temp}°C",
        #         fontsize=5,
        #     )
        # else:
        #     ax.set_title(
        #         f"Normalized Peak Area for m/z {m_z_start} to {m_z_end} for {tga_type} TGA-MS data"
        #     )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=5, direction="in")
        for ms_path, label, color, alpha in zip(
            self.ms_data_path, self.labels, self.colors, alpha_list
        ):
            ms_data = pd.read_csv(
                self.data_dir / ms_path,
                encoding="iso-8859-1",
                on_bad_lines="skip",
                skiprows=29,
            )
            ms_data = ms_data.iloc[
                (ms_data["Time/min"] - initial_time)
                .abs()
                .argsort()[: (ms_data["Time/min"] - end_time).abs().argsort()[0]]
            ]
            # find index at which the m/z starts and ends
            m_z_start_idx: int = ms_data.columns.get_loc(f"QMID(s:1|m:{m_z_start})/A")
            m_z_end_idx: int = ms_data.columns.get_loc(f"QMID(s:1|m:{m_z_end})/A")
            peak_area = ms_data.iloc[:, m_z_start_idx:m_z_end_idx].sum(axis=0)
            if normalize:
                peak_area = peak_area / peak_area.max()
            # ax.plot(
            #     range(m_z_start, m_z_end),
            #     peak_area,
            #     label=label,
            #     color=color,
            #     marker="o",
            #     markersize=0.5,
            #     alpha=0.6,
            #     linewidth=0.5,
            # )
            ax.vlines(
                range(m_z_start, m_z_end),
                0,
                peak_area,
                colors=color,
                alpha=alpha,
                linewidth=0.5,
                label=label,
            )
        # label all the points with m/z that are above 10% of the max peak area
        for i, txt in enumerate(peak_area):
            if txt > 0.1 * peak_area.max():
                ax.annotate(
                    f"{range(m_z_start, m_z_end)[i]}",
                    (range(m_z_start, m_z_end)[i], peak_area[i]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=5,
                )

        ax.legend(fontsize=5)
        plt.savefig(
            self.result_dir
            / f"{self.result_name}ms_peak_area_{m_z_start}_{m_z_end}_{isothermal_temp}.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir
            / f"{self.result_name}ms_peak_area_{m_z_start}_{m_z_end}_{isothermal_temp}.eps",
            format="eps",
            dpi=400,
        )

    def calculate_onset_t(self, tga_data: pd.DataFrame, xlim: tuple):
        """Calculate onset temperature using two linear extrapolations (one from the initial baseline, and another from the greatest gradient)

        Args:
            tga_data (pd.DataFrame): _description_
        """
        # Filter for xlim windoww
        tga_data = tga_data[tga_data["Temp"].between(xlim[0], xlim[1] - 5)]
        tga_data.reset_index(inplace=True)
        # 1. Calculate initial baseline (before significant mass loss)
        # Use data points before 1% mass loss for baseline
        baseline_data = tga_data[tga_data["Mass loss/pct"] > 99]
        if len(baseline_data) < 1:
            # If not enough points, use first 10% of data
            baseline_data = tga_data.iloc[: max(2, len(tga_data) // 10)]

        baseline_slope, baseline_intercept, _, _, _ = linregress(
            baseline_data["Temp"], baseline_data["Mass loss/pct"]
        )

        # 2. Find the region with greatest gradient (steepest mass loss)
        # Calculate derivative (gradient) of mass loss with respect to temperature
        temp_diff = tga_data["Temp"].diff()
        mass_diff = tga_data["Mass loss/pct"].diff()
        gradient = mass_diff / temp_diff

        # Find the index with maximum gradient (steepest slope)
        # Use rolling window to smooth and find sustained high gradient
        window_size = min(10, len(gradient) // 20)  # Adaptive window size
        if window_size < 2:
            window_size = 2

        gradient_smooth = gradient.rolling(window=window_size, center=True).mean()
        max_gradient_idx = gradient_smooth.idxmin()  # most negative

        # Define region around maximum gradient for linear fit
        # Use points within ±20% of data length around max gradient point
        fit_range = max(5, len(tga_data) // 10)
        start_idx = max(0, max_gradient_idx - fit_range)
        end_idx = min(len(tga_data), max_gradient_idx + fit_range)

        gradient_region = tga_data.iloc[start_idx:end_idx]

        # Fit linear regression to the steepest gradient region
        gradient_slope, gradient_intercept, _, _, _ = linregress(
            gradient_region["Temp"], gradient_region["Mass loss/pct"]
        )

        # 3. Calculate onset temperature as intersection of the two lines
        # baseline: y = baseline_slope * x + baseline_intercept
        # gradient: y = gradient_slope * x + gradient_intercept
        # Intersection: baseline_slope * x + baseline_intercept = gradient_slope * x + gradient_intercept
        # Solving for x: x = (gradient_intercept - baseline_intercept) / (baseline_slope - gradient_slope)

        if abs(baseline_slope - gradient_slope) < 1e-10:
            # Lines are parallel, return temperature at maximum gradient
            onset_temp = tga_data.iloc[max_gradient_idx]["Temp"]
        else:
            onset_temp = (gradient_intercept - baseline_intercept) / (
                baseline_slope - gradient_slope
            )

        return (
            baseline_slope,
            baseline_intercept,
            gradient_slope,
            gradient_intercept,
            onset_temp,
        )

    def plot_onset_temperature_vs_mn(
        self, summary_dir, ylim=(315, 375), xlim=(-10, 100)
    ):
        """
        Plot dynamic onset temperature vs molecular weight for PS samples
        with and without SCF3 functionalization.

        Parameters:
        summary_dir (str): Directory containing the mn_summary.csv file

        Returns:
        None: Displays the plot
        """

        # Read the CSV file
        csv_path = self.data_dir / summary_dir
        df = pd.read_csv(csv_path)

        # Separate data into functionalized and non-functionalized samples
        df_pristine = df[~df["name"].str.contains("SCF3", na=False)]
        df_functionalized = df[df["name"].str.contains("SCF3", na=False)]

        # Clean data for onset temperature (remove NaN values)
        df_pristine_onset = df_pristine.dropna(subset=["mn", "dynamic_onset_t"])
        df_func_onset = df_functionalized.dropna(subset=["mn", "dynamic_onset_t"])

        # Create the plot with matching style
        fig, ax = plt.subplots(1, figsize=(3.5, 3))
        plt.subplots_adjust(hspace=0.5)

        # Aesthetics matching plot_tga_isothermal
        ax.set_xlabel("Molecular Weight (Mn) [kg/mol]", fontsize=10)
        ax.set_ylabel("Dynamic Onset Temperature [°C]", fontsize=10)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8, direction="in")

        # Plot onset temperature data
        ax.scatter(
            df_pristine_onset["mn"],
            df_pristine_onset["dynamic_onset_t"],
            color="#000000",
            marker="o",
            alpha=0.7,
            label="Pristine PS",
            linewidth=1,
        )

        # Plot line of best fit for pristine samples
        z = np.polyfit(df_pristine_onset["mn"], df_pristine_onset["dynamic_onset_t"], 1)
        p = np.poly1d(z)
        ax.plot(
            df_pristine_onset["mn"],
            p(df_pristine_onset["mn"]),
            "--",
            color="#000000",
            alpha=0.7,
            linewidth=1,
        )

        ax.scatter(
            df_func_onset["mn"],
            df_func_onset["dynamic_onset_t"],
            color="#F4BD14",
            marker="o",
            alpha=0.7,
            label="PS-SCF3",
            linewidth=1,
        )
        # Plot line of best fit for functionalized samples
        z = np.polyfit(df_func_onset["mn"], df_func_onset["dynamic_onset_t"], 1)
        p = np.poly1d(z)
        ax.plot(
            df_func_onset["mn"],
            p(df_func_onset["mn"]),
            "--",
            color="#F4BD14",
            alpha=0.7,
            linewidth=1,
        )

        # Legend with matching fontsize
        ax.legend(fontsize=10)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(
            self.result_dir / f"onset_temperature_vs_mn.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir / f"onset_temperature_vs_mn.eps",
            format="eps",
            dpi=400,
        )

    def plot_depoly_temperature_vs_mn(
        self, summary_dir, ylim=(250, 350), xlim=(-10, 100)
    ):
        """
        Plot dynamic depolymerization temperature vs molecular weight for PS samples
        with and without SCF3 functionalization.

        Parameters:
        summary_dir (str): Directory containing the mn_summary.csv file

        Returns:
        None: Displays the plot
        """

        # Read the CSV file
        csv_path = self.data_dir / summary_dir
        df = pd.read_csv(csv_path)

        # Separate data into functionalized and non-functionalized samples
        df_pristine = df[~df["name"].str.contains("SCF3", na=False)]
        df_functionalized = df[df["name"].str.contains("SCF3", na=False)]

        # Clean data for onset temperature (remove NaN values)
        df_pristine_onset = df_pristine.dropna(
            subset=["mn", "dynamic_depolymerization_t"]
        )
        df_func_onset = df_functionalized.dropna(
            subset=["mn", "dynamic_depolymerization_t"]
        )

        # Create the plot with matching style
        fig, ax = plt.subplots(1, figsize=(3.5, 3))
        plt.subplots_adjust(hspace=0.5)

        # Aesthetics matching plot_tga_isothermal
        ax.set_xlabel("Molecular Weight (Mn) [kg/mol]", fontsize=10)
        ax.set_ylabel("Dynamic Depolymerization Temperature [°C]", fontsize=10)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8, direction="in")

        # Plot onset temperature data
        ax.scatter(
            df_pristine_onset["mn"],
            df_pristine_onset["dynamic_depolymerization_t"],
            color="#000000",
            marker="o",
            alpha=0.7,
            label="Pristine PS",
            linewidth=1,
        )

        ax.scatter(
            df_func_onset["mn"],
            df_func_onset["dynamic_depolymerization_t"],
            color="#F4BD14",
            marker="o",
            alpha=0.7,
            label="PS-SCF3",
            linewidth=1,
        )

        # Legend with matching fontsize
        ax.legend(fontsize=10)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(
            self.result_dir / f"depolymerization_temperature_vs_mn.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir / f"depolymerization_temperature_vs_mn.eps",
            format="eps",
            dpi=400,
        )

    def plot_mass_loss_vs_mn(self, summary_dir, ylim=(20, 80), xlim=(-10, 100)):
        """
        Plot isothermal mass loss vs molecular weight for PS samples
        with and without SCF3 functionalization.

        Parameters:
        summary_dir (str): Directory containing the mn_summary.csv file

        Returns:
        None: Displays the plot
        """

        # Read the CSV file
        csv_path = self.data_dir / summary_dir
        df = pd.read_csv(csv_path)

        # Separate data into functionalized and non-functionalized samples
        df_pristine = df[~df["name"].str.contains("SCF3", na=False)]
        df_functionalized = df[df["name"].str.contains("SCF3", na=False)]

        # Clean data for mass loss (remove NaN values)
        df_pristine_mass = df_pristine.dropna(
            subset=["mn", "isothermal_mass_loss_after_1200mins"]
        )
        df_func_mass = df_functionalized.dropna(
            subset=["mn", "isothermal_mass_loss_after_1200mins"]
        )

        # Create the plot with matching style
        fig, ax = plt.subplots(1, figsize=(3.5, 3))
        plt.subplots_adjust(hspace=0.5)

        # Aesthetics matching plot_tga_isothermal
        ax.set_xlabel("Molecular Weight (Mn) [kg/mol]", fontsize=10)
        ax.set_ylabel("Isothermal Mass Loss after 1200 mins [%]", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8, direction="in")
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

        # Plot mass loss data
        ax.scatter(
            df_pristine_mass["mn"],
            df_pristine_mass["isothermal_mass_loss_after_1200mins"],
            color="#000000",
            marker="o",
            alpha=0.7,
            label="Pristine PS",
            linewidth=1,
        )
        # Plot line of best fit
        z = np.polyfit(
            df_pristine_mass["mn"],
            df_pristine_mass["isothermal_mass_loss_after_1200mins"],
            1,
        )
        p = np.poly1d(z)
        ax.plot(
            df_pristine_mass["mn"],
            p(df_pristine_mass["mn"]),
            "--",
            color="#000000",
            alpha=0.7,
            linewidth=1,
        )

        ax.scatter(
            df_func_mass["mn"],
            df_func_mass["isothermal_mass_loss_after_1200mins"],
            color="#F4BD14",
            marker="o",
            alpha=0.7,
            label="PS-SCF3",
            linewidth=1,
        )
        # Plot line of best fit
        z = np.polyfit(
            df_func_mass["mn"], df_func_mass["isothermal_mass_loss_after_1200mins"], 1
        )
        p = np.poly1d(z)
        ax.plot(
            df_func_mass["mn"],
            p(df_func_mass["mn"]),
            "--",
            color="#F4BD14",
            alpha=0.7,
            linewidth=1,
        )

        # Legend with matching fontsize
        ax.legend(fontsize=10)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(
            self.result_dir / f"summary_plots_mass_loss_vs_mn.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir / f"summary_plots_mass_loss_vs_mn.eps",
            format="eps",
            dpi=400,
        )

    def plot_onset_temperature_vs_percent_functionalization(
        self, summary_dir, ylim=(310, 360), xlim=(-0.5, 4)
    ):
        """∂
        Plot dynamic onset temperature vs percent functionalization for PS samples
        with and without SCF3 functionalization.

        Parameters:
        summary_dir (str): Directory containing the mn_summary.csv file

        Returns:
        None: Displays the plot
        """

        # Read the CSV file
        csv_path = self.data_dir / summary_dir
        df = pd.read_csv(csv_path)

        # Separate data into functionalized and non-functionalized samples
        df_pristine = df[~df["name"].str.contains("SCF3", na=False)]
        df_functionalized = df[df["name"].str.contains("SCF3", na=False)]

        # Clean data for onset temperature (remove NaN values)
        df_pristine_onset = df_pristine.dropna(
            subset=["percent_functionalization", "dynamic_onset_t"]
        )
        df_func_onset = df_functionalized.dropna(
            subset=["percent_functionalization", "dynamic_onset_t"]
        )

        # Create the plot with matching style
        fig, ax = plt.subplots(1, figsize=(3.5, 3))
        plt.subplots_adjust(hspace=0.5)

        # Aesthetics matching plot_tga_isothermal
        ax.set_xlabel("Percent Functionalization [%]", fontsize=10)
        ax.set_ylabel("Dynamic Onset Temperature [°C]", fontsize=10)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8, direction="in")

        # Plot onset temperature data
        ax.scatter(
            df_pristine_onset["percent_functionalization"],
            df_pristine_onset["dynamic_onset_t"],
            color="#000000",
            marker="^",
            alpha=0.7,
            label="Pristine PS",
            linewidth=1,
        )

        ax.scatter(
            df_func_onset["percent_functionalization"],
            df_func_onset["dynamic_onset_t"],
            color="#F4BD14",
            marker="^",
            alpha=0.7,
            label="PS-SCF3",
            linewidth=1,
        )

        # Legend with matching fontsize
        ax.legend(fontsize=10)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(
            self.result_dir / f"onset_temperature_vs_percent_functionalization.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir / f"onset_temperature_vs_percent_functionalization.eps",
            format="eps",
            dpi=400,
        )

    def plot_depoly_temperature_vs_percent_functionalization(
        self, summary_dir, ylim=(250, 300), xlim=(-0.5, 4)
    ):
        """∂
        Plot dynamic depolymerization temperature vs percent functionalization for PS samples
        with and without SCF3 functionalization.

        Parameters:
        summary_dir (str): Directory containing the mn_summary.csv file

        Returns:
        None: Displays the plot
        """

        # Read the CSV file
        csv_path = self.data_dir / summary_dir
        df = pd.read_csv(csv_path)

        # Separate data into functionalized and non-functionalized samples
        df_pristine = df[~df["name"].str.contains("SCF3", na=False)]
        df_functionalized = df[df["name"].str.contains("SCF3", na=False)]

        # Clean data for onset temperature (remove NaN values)
        df_pristine_onset = df_pristine.dropna(
            subset=["percent_functionalization", "dynamic_depolymerization_t"]
        )
        df_func_onset = df_functionalized.dropna(
            subset=["percent_functionalization", "dynamic_depolymerization_t"]
        )

        # Create the plot with matching style
        fig, ax = plt.subplots(1, figsize=(3.5, 3))
        plt.subplots_adjust(hspace=0.5)

        # Aesthetics matching plot_tga_isothermal
        ax.set_xlabel("Percent Functionalization [%]", fontsize=10)
        ax.set_ylabel("Dynamic Depolymerization Temperature [°C]", fontsize=10)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8, direction="in")

        # Plot onset temperature data
        ax.scatter(
            df_pristine_onset["percent_functionalization"],
            df_pristine_onset["dynamic_depolymerization_t"],
            color="#000000",
            marker="^",
            alpha=0.7,
            label="Pristine PS",
            linewidth=1,
        )

        ax.scatter(
            df_func_onset["percent_functionalization"],
            df_func_onset["dynamic_depolymerization_t"],
            color="#F4BD14",
            marker="^",
            alpha=0.7,
            label="PS-SCF3",
            linewidth=1,
        )

        # Legend with matching fontsize
        ax.legend(fontsize=10)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(
            self.result_dir / f"depoly_temperature_vs_percent_functionalization.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir / f"depoly_temperature_vs_percent_functionalization.eps",
            format="eps",
            dpi=400,
        )

    def plot_mass_loss_vs_percent_functionalization(
        self, summary_dir, ylim=(40, 80), xlim=(-0.5, 4)
    ):
        """
        Plot isothermal mass loss vs percent functionalization for PS samples
        with and without SCF3 functionalization.

        Parameters:
        summary_dir (str): Directory containing the mn_summary.csv file

        Returns:
        None: Displays the plot
        """

        # Read the CSV file
        csv_path = self.data_dir / summary_dir
        df = pd.read_csv(csv_path)

        # Separate data into functionalized and non-functionalized samples
        df_pristine = df[~df["name"].str.contains("SCF3", na=False)]
        df_functionalized = df[df["name"].str.contains("SCF3", na=False)]

        # Clean data for mass loss (remove NaN values)
        df_pristine_mass = df_pristine.dropna(
            subset=["percent_functionalization", "isothermal_mass_loss_after_1200mins"]
        )
        df_func_mass = df_functionalized.dropna(
            subset=["percent_functionalization", "isothermal_mass_loss_after_1200mins"]
        )

        # Create the plot with matching style
        fig, ax = plt.subplots(1, figsize=(3.5, 3))
        plt.subplots_adjust(hspace=0.5)

        # Aesthetics matching plot_tga_isothermal
        ax.set_xlabel("Percent Functionalization [%]", fontsize=10)
        ax.set_ylabel("Isothermal Mass Loss after 1200 mins [%]", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8, direction="in")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_yticks(range(int(ylim[0]), int(ylim[1]) + 1, 10))

        # Plot mass loss data
        ax.scatter(
            df_pristine_mass["percent_functionalization"],
            df_pristine_mass["isothermal_mass_loss_after_1200mins"],
            color="#000000",
            marker="^",
            alpha=0.7,
            label="Pristine PS",
            linewidth=1,
        )

        ax.scatter(
            df_func_mass["percent_functionalization"],
            df_func_mass["isothermal_mass_loss_after_1200mins"],
            color="#F4BD14",
            marker="^",
            alpha=0.7,
            label="PS-SCF3",
            linewidth=1,
        )

        # Legend with matching fontsize
        ax.legend(fontsize=10)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(
            self.result_dir
            / f"summary_plots_mass_loss_vs_percent_functionalization.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir
            / f"summary_plots_mass_loss_vs_percent_functionalization.eps",
            format="eps",
            dpi=400,
        )

    def plot_overlay_isothermal(
        self,
        virgin_ps_data_paths,
        scf3_ps_data_paths,
        isothermal_temp: float = 300,
        target_mass: int = 104,
        xlim: tuple = (0, 1200),
        ylim: tuple = (0, 100),
        initial_correction_time: int = 50,
        uncertainty: bool = False,
        time_for_mass_difference: float = 1200,
    ):
        """
        Plots an overlay of all the virgin PS and PS-SCF3 isothermal graphs.

        Args:
            virgin_ps_data_paths (list): List of tuples [(tga_path, ms_path, label), ...]
            scf3_ps_data_paths (list): List of tuples [(tga_path, ms_path, label), ...]
            isothermal_temp (float): Temperature for isothermal analysis
            target_mass (int): Target mass for MS analysis
            xlim (tuple): X-axis limits
            ylim (tuple): Y-axis limits
            initial_correction_time (int): Time correction for initial period
            uncertainty (bool): Whether to show uncertainty bands
            time_for_mass_difference (float): Time point for mass difference calculation
        """

        fig, ax = plt.subplots(1, figsize=(4, 3.25))
        plt.subplots_adjust(hspace=0.5)

        # Aesthetics
        ax.set_xlabel("Time (min)", fontsize=10)
        ax.set_ylabel("Mass (%)", fontsize=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=10, direction="in")

        # Set limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        # Generate different shades of #F4BD14 for SCF3 samples
        base_color = "#F4BD14"
        scf3_colors = []
        num_scf3 = len(scf3_ps_data_paths)

        if num_scf3 > 0:
            # Create variations by adjusting saturation and lightness
            import matplotlib.colors as mcolors

            base_rgb = mcolors.hex2color(base_color)
            base_hsv = mcolors.rgb_to_hsv(base_rgb)

            for i in range(num_scf3):
                # Vary the saturation and value to create different shades
                saturation_factor = 0.6 + (0.4 * i / max(1, num_scf3 - 1))  # 0.6 to 1.0
                value_factor = 0.7 + (0.3 * i / max(1, num_scf3 - 1))  # 0.7 to 1.0

                new_hsv = [
                    base_hsv[0],
                    base_hsv[1] * saturation_factor,
                    base_hsv[2] * value_factor,
                ]
                new_rgb = mcolors.hsv_to_rgb(new_hsv)
                scf3_colors.append(mcolors.rgb2hex(new_rgb))
        # Reverse scf3_colors order
        scf3_colors.reverse()

        # Gray colors for virgin PS samples
        virgin_colors = ["#808080", "#000000", "#606060", "#909090", "#707070"]

        mass_difference_at_time = []

        # Plot virgin PS samples in gray
        # for i, (tga_path, label) in enumerate(virgin_ps_data_paths):
        #     color = virgin_colors[i % len(virgin_colors)]

        #     # Read and process TGA data
        #     tga_data = pd.read_csv(
        #         self.data_dir / tga_path, encoding="iso-8859-1", on_bad_lines="skip"
        #     )
        #     initial_mass: float = float(tga_data.at[17, tga_data.columns[1]])
        #     num_skiprows: int = _find_empty_row(tga_data)
        #     tga_data = pd.read_csv(
        #         self.data_dir / tga_path,
        #         encoding="iso-8859-1",
        #         on_bad_lines="skip",
        #         skiprows=num_skiprows,
        #     )

        #     # Preprocess data
        #     tga_data, ms_data = self.preprocess(
        #         tga_data, None, initial_correction_time, "Time"
        #     )

        #     mass_difference_at_time.append(
        #         100 - self.get_mass_at_time(time_for_mass_difference, tga_data)
        #     )

        #     # Plot TGA data
        #     ax.plot(
        #         tga_data["Time/min"] - initial_correction_time,
        #         tga_data["Mass loss/pct"],
        #         label=f"Virgin PS - {label}",
        #         color=color,
        #         linewidth=1.5,
        #         linestyle="-",
        #     )

        # Plot PS-SCF3 samples in shades of #F4BD14
        for i, (tga_path, label) in enumerate(scf3_ps_data_paths):
            color = scf3_colors[i % len(scf3_colors)]

            # Read and process TGA data
            tga_data = pd.read_csv(
                self.data_dir / tga_path, encoding="iso-8859-1", on_bad_lines="skip"
            )
            initial_mass: float = float(tga_data.at[17, tga_data.columns[1]])
            num_skiprows: int = _find_empty_row(tga_data)
            tga_data = pd.read_csv(
                self.data_dir / tga_path,
                encoding="iso-8859-1",
                on_bad_lines="skip",
                skiprows=num_skiprows,
            )

            # Preprocess data
            tga_data, ms_data = self.preprocess(
                tga_data, None, initial_correction_time, "Time"
            )

            mass_difference_at_time.append(
                100 - self.get_mass_at_time(time_for_mass_difference, tga_data)
            )

            # Plot TGA data
            ax.plot(
                tga_data["Time/min"] - initial_correction_time,
                tga_data["Mass loss/pct"],
                label=f"PS-SCF3 (Mn={label})",
                color=color,
                linewidth=1,
                linestyle="-",
            )

        # Add reference line at 100% mass
        ax.axhline(
            y=100,
            color="r",
            linestyle="--",
            linewidth=0.5,
            label="100% Mass",
        )

        # Add legends
        ax.legend(fontsize=10, loc="best")

        # Save the plots
        plt.savefig(
            self.result_dir
            / f"{self.result_name}overlay_isothermal_{isothermal_temp}_{target_mass}m_z.png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.savefig(
            self.result_dir
            / f"{self.result_name}overlay_isothermal_{isothermal_temp}_{target_mass}m_z.eps",
            format="eps",
            dpi=400,
            bbox_inches="tight",
        )

    def plot_overlay_dynamic(
        self,
        virgin_ps_data_paths,
        scf3_ps_data_paths,
        xlim: tuple = (250, 400),
        ylim: tuple = (0, 105),
        initial_correction_temp: int = 200,
    ):
        """
        Plots an overlay of all the virgin PS and PS-SCF3 dynamic TGA graphs.

        Args:
            virgin_ps_data_paths (list): List of tuples [(tga_path, label), ...]
            scf3_ps_data_paths (list): List of tuples [(tga_path, label), ...]
            xlim (tuple): X-axis limits
            ylim (tuple): Y-axis limits
            initial_correction_temp (int): Temperature correction for initial period
        """

        fig, ax = plt.subplots(1, figsize=(4, 3.25))
        plt.subplots_adjust(hspace=0.5)

        # Aesthetics
        ax.set_xlabel("Temp ($^{o}$C)", fontsize=10)
        ax.set_ylabel("Mass (%)", fontsize=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=10, direction="in")

        # Set limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        # Generate different shades of #F4BD14 for SCF3 samples
        base_color = "#F4BD14"
        scf3_colors = []
        num_scf3 = len(scf3_ps_data_paths)

        if num_scf3 > 0:
            # Create variations by adjusting saturation and lightness
            import matplotlib.colors as mcolors

            base_rgb = mcolors.hex2color(base_color)
            base_hsv = mcolors.rgb_to_hsv(base_rgb)

            for i in range(num_scf3):
                # Vary the saturation and value to create different shades
                saturation_factor = 0.6 + (0.4 * i / max(1, num_scf3 - 1))  # 0.6 to 1.0
                value_factor = 0.7 + (0.3 * i / max(1, num_scf3 - 1))  # 0.7 to 1.0

                new_hsv = [
                    base_hsv[0],
                    base_hsv[1] * saturation_factor,
                    base_hsv[2] * value_factor,
                ]
                new_rgb = mcolors.hsv_to_rgb(new_hsv)
                scf3_colors.append(mcolors.rgb2hex(new_rgb))
        # Reverse scf3_colors order
        scf3_colors.reverse()

        # # Gray colors for virgin PS samples
        # virgin_colors = ["#808080", "#000000", "#606060", "#909090", "#707070"]

        # # Plot virgin PS samples in gray
        # for i, (tga_path, label) in enumerate(virgin_ps_data_paths):
        #     color = virgin_colors[i % len(virgin_colors)]

        #     # Read and process TGA data
        #     tga_data = pd.read_csv(
        #         self.data_dir / tga_path,
        #         encoding="iso-8859-1",
        #         on_bad_lines="skip",
        #         skiprows=37,
        #     )

        #     # Preprocess data
        #     tga_data, ms_data = self.preprocess(
        #         tga_data, None, initial_correction_temp, "Temp"
        #     )

        #     # Plot TGA data
        #     ax.plot(
        #         tga_data["Temp"],
        #         tga_data["Mass loss/pct"],
        #         label=f"Virgin PS - {label}",
        #         color=color,
        #         linewidth=1.5,
        #         linestyle="-",
        #     )

        # Plot PS-SCF3 samples in shades of #F4BD14
        for i, (tga_path, label) in enumerate(scf3_ps_data_paths):
            color = scf3_colors[i % len(scf3_colors)]

            # Read and process TGA data
            tga_data = pd.read_csv(
                self.data_dir / tga_path,
                encoding="iso-8859-1",
                on_bad_lines="skip",
                skiprows=37,
            )

            # Preprocess data
            tga_data, ms_data = self.preprocess(
                tga_data, None, initial_correction_temp, "Temp"
            )

            # Plot TGA data
            ax.plot(
                tga_data["Temp"],
                tga_data["Mass loss/pct"],
                label=f"PS-SCF3 (Mn={label})",
                color=color,
                linewidth=1,
                linestyle="-",
            )

        # Add legends
        ax.legend(fontsize=10, loc="best")

        # Save the plots
        plt.savefig(
            self.result_dir / f"{self.result_name}overlay_dynamic_tga.png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.savefig(
            self.result_dir / f"{self.result_name}overlay_dynamic_tga.eps",
            format="eps",
            dpi=400,
            bbox_inches="tight",
        )

    def plot_tga_isothermal_rate_constant(
        self,
        isothermal_temp: float,
        target_mass: int = 104,
        xlim: tuple = (0, 1200),
        ylim: tuple = (0, 100),
        initial_correction_time: int = 50,
        fit_start_time: float = 0,
        fit_end_time: float = 1200,
    ):
        """
        Plot TGA isothermal data with 1st order reaction model fitting.

        For a 1st order reaction with complete degradation, mass follows:
        m(t) = m_0 * exp(-k*t)

        where:
        - m(t) is the mass percentage at time t
        - m_0 is the initial mass percentage
        - k is the rate constant (1/min)
        - t is time (min)
        - m_∞ = 0 (asymptote at zero mass, complete degradation)

        Args:
            isothermal_temp: Temperature for isothermal analysis
            target_mass: Target mass for MS analysis
            xlim: X-axis limits
            ylim: Y-axis limits
            initial_correction_time: Time correction for initial period
            fit_start_time: Start time for fitting (relative to corrected time)
            fit_end_time: End time for fitting (relative to corrected time)
        """

        def first_order_model(t, m_0, k):
            """First order reaction model assuming complete degradation (m_∞ = 0)."""
            return m_0 * np.exp(-k * t)

        fig, ax = plt.subplots(1, figsize=(6, 4))
        plt.subplots_adjust(hspace=0.5)

        # Aesthetics
        ax.set_xlabel("Time (min)", fontsize=10)
        ax.set_ylabel("Mass (%)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8, direction="in")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

        rate_constants = []
        r2_scores = []

        for tga_path, ms_path, label, color in zip(
            self.tga_data_path, self.ms_data_path, self.labels, self.colors
        ):
            # Read TGA data
            tga_data = pd.read_csv(
                self.data_dir / tga_path, encoding="iso-8859-1", on_bad_lines="skip"
            )
            initial_mass: float = float(tga_data.at[17, tga_data.columns[1]])
            num_skiprows: int = _find_empty_row(tga_data)
            tga_data = pd.read_csv(
                self.data_dir / tga_path,
                encoding="iso-8859-1",
                on_bad_lines="skip",
                skiprows=num_skiprows,
            )

            # Preprocess data
            tga_data, ms_data = self.preprocess(
                tga_data, None, initial_correction_time, "Time"
            )

            # Adjust time to start from 0
            time_data = tga_data["Time/min"].values - initial_correction_time
            mass_data = tga_data["Mass loss/pct"].values

            # Determine fitting range
            if fit_end_time is None:
                fit_end_time = time_data[-1]

            # Filter data for fitting
            fit_mask = (time_data >= fit_start_time) & (time_data <= fit_end_time)
            time_fit = time_data[fit_mask]
            mass_fit = mass_data[fit_mask]

            # Initial parameter guesses
            m_0_guess = mass_fit[0]
            k_guess = 0.001  # Initial guess for rate constant

            try:
                # Fit with m_∞ = 0 (complete degradation model)
                popt, pcov = curve_fit(
                    first_order_model,
                    time_fit,
                    mass_fit,
                    p0=[m_0_guess, k_guess],
                    maxfev=10000,
                    bounds=(
                        [95, 0],  # Lower bounds: m_0>=95, k>=0
                        [100, 1],  # Upper bounds: m_0<=100, k<=1
                    ),
                )
                m_0_fit, k_fit = popt

                # Calculate R² score
                mass_pred = first_order_model(time_fit, *popt)
                r2 = r2_score(mass_fit, mass_pred)

                rate_constants.append(k_fit)
                r2_scores.append(r2)

                # Plot experimental data
                ax.plot(
                    time_data,
                    mass_data,
                    label=f"{label}",
                    color=color,
                    linewidth=1,
                    alpha=0.7,
                )

                # Plot fitted curve
                time_smooth = np.linspace(time_fit[0], time_fit[-1], 500)
                mass_smooth = first_order_model(time_smooth, *popt)

                ax.plot(
                    time_smooth,
                    mass_smooth,
                    label=f"{label} fit (k={k_fit:.2e} min$^{{-1}}$, R²={r2:.4f})",
                    color=color,
                    linewidth=1.5,
                    linestyle="--",
                )

                # Calculate standard errors from covariance matrix
                perr = np.sqrt(np.diag(pcov))

                print(f"\n{label}:")
                print(f"  Rate constant (k): {k_fit:.6e} ± {perr[1]:.6e} min^-1")
                print(f"  Initial mass (m_0): {m_0_fit:.2f} ± {perr[0]:.2f}%")
                print(f"  Final mass (m_∞): 0.00% (complete degradation)")
                print(f"  Total mass loss (m_0 - m_∞): {m_0_fit:.2f}%")
                print(f"  Effective degradation rate: {k_fit * m_0_fit:.6e}")
                print(f"  R² score: {r2:.6f}")
                print(f"  Half-life: {np.log(2)/k_fit:.2f} min")

            except Exception as e:
                print(f"Warning: Fitting failed for {label}: {str(e)}")
                # Still plot the raw data
                ax.plot(
                    time_data,
                    mass_data,
                    label=f"{label} (fit failed)",
                    color=color,
                    linewidth=1,
                )

        ax.axhline(
            y=100,
            color="r",
            linestyle="--",
            linewidth=0.5,
            label="100% Mass",
        )

        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()

        # Save plots
        plt.savefig(
            self.result_dir
            / f"{self.result_name}tga_isothermal_{isothermal_temp}_{target_mass}m_z_rate_constant.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir
            / f"{self.result_name}tga_isothermal_{isothermal_temp}_{target_mass}m_z_rate_constant.eps",
            format="eps",
            dpi=400,
        )

        return rate_constants, r2_scores

    def plot_tga_isothermal_kinetic_analysis(
        self,
        isothermal_temp: float,
        target_mass: int = 104,
        xlim: tuple = (0, 1200),
        ylim: tuple = (0, 100),
        initial_correction_time: int = 50,
        fit_start_time: float = 0,
        fit_end_time: float = 1200,
    ):
        """
        Comprehensive kinetic analysis comparing 1st and 2nd order models.
        Assumes complete degradation (m_∞ = 0).

        Creates three plots:
        1. Experimental data with both 1st and 2nd order fits
        2. Linearized 1st order plot: log(m) vs time
        3. Linearized 2nd order plot: 1/m vs time

        Args:
            isothermal_temp: Temperature for isothermal analysis
            target_mass: Target mass for MS analysis
            xlim: X-axis limits for main plot
            ylim: Y-axis limits for main plot
            initial_correction_time: Time correction for initial period
            fit_start_time: Start time for fitting (relative to corrected time)
            fit_end_time: End time for fitting (relative to corrected time)
        """

        def first_order_model(t, m_0, k):
            """First order reaction model: m(t) = m_0 * exp(-k*t)"""
            return m_0 * np.exp(-k * t)

        def second_order_model(t, m_0, k):
            """Second order reaction model: m(t) = m_0 / (1 + k*m_0*t)"""
            return m_0 / (1 + k * m_0 * t)

        # Create figure with 3 subplots
        fig = plt.figure(figsize=(15, 4))
        ax1 = plt.subplot(1, 3, 1)  # Fitted curves
        ax2 = plt.subplot(1, 3, 2)  # 1st order linearized
        ax3 = plt.subplot(1, 3, 3)  # 2nd order linearized

        # Aesthetics for all subplots
        for ax in [ax1, ax2, ax3]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="major", labelsize=8, direction="in")

        # Main plot aesthetics
        ax1.set_xlabel("Time (min)", fontsize=10)
        ax1.set_ylabel("Mass (%)", fontsize=10)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_yticks([0, 20, 40, 60, 80, 100])
        ax1.set_title("Experimental Data with Fits", fontsize=10)

        # Linearized plot aesthetics
        ax2.set_xlabel("Time (min)", fontsize=10)
        ax2.set_ylabel("log(m)", fontsize=10)
        ax2.set_title("1st Order: log(m) vs t", fontsize=10)

        ax3.set_xlabel("Time (min)", fontsize=10)
        ax3.set_ylabel("1/m", fontsize=10)
        ax3.set_title("2nd Order: 1/m vs t", fontsize=10)

        results = []

        for tga_path, ms_path, label, color in zip(
            self.tga_data_path, self.ms_data_path, self.labels, self.colors
        ):
            # Read TGA data
            tga_data = pd.read_csv(
                self.data_dir / tga_path, encoding="iso-8859-1", on_bad_lines="skip"
            )
            num_skiprows: int = _find_empty_row(tga_data)
            tga_data = pd.read_csv(
                self.data_dir / tga_path,
                encoding="iso-8859-1",
                on_bad_lines="skip",
                skiprows=num_skiprows,
            )

            # Preprocess data
            tga_data, _ = self.preprocess(
                tga_data, None, initial_correction_time, "Time"
            )

            # Adjust time to start from 0
            time_data = tga_data["Time/min"].values - initial_correction_time
            mass_data = tga_data["Mass loss/pct"].values

            # Filter data for fitting
            if fit_end_time is None:
                fit_end_time = time_data[-1]

            fit_mask = (time_data >= fit_start_time) & (time_data <= fit_end_time)
            time_fit = time_data[fit_mask]
            mass_fit = mass_data[fit_mask]

            # Initial parameter guesses
            m_0_guess = mass_fit[0]
            k_guess_first = 0.001
            k_guess_second = 0.00001

            # ============ FIRST ORDER FITTING ============
            try:
                popt_1st, pcov_1st = curve_fit(
                    first_order_model,
                    time_fit,
                    mass_fit,
                    p0=[m_0_guess, k_guess_first],
                    maxfev=10000,
                    bounds=([95, 0], [100, 1]),
                )
                m_0_1st, k_1st = popt_1st
                mass_pred_1st = first_order_model(time_fit, *popt_1st)
                r2_1st = r2_score(mass_fit, mass_pred_1st)
                perr_1st = np.sqrt(np.diag(pcov_1st))

                # Plot 1st order fit on main plot
                time_smooth = np.linspace(time_fit[0], time_fit[-1], 500)
                mass_smooth_1st = first_order_model(time_smooth, *popt_1st)
                ax1.plot(
                    time_smooth,
                    mass_smooth_1st,
                    label=f"{label} (1st order, R²={r2_1st:.4f})",
                    color=color,
                    linewidth=1.5,
                    linestyle="--",
                )

                # Linearized 1st order plot: log(m) vs t
                valid_mask_1st = mass_fit > 0.01
                if np.sum(valid_mask_1st) > 5:
                    time_lin_1st = time_fit[valid_mask_1st]
                    log_term_1st = np.log(mass_fit[valid_mask_1st])

                    # Linear fit to get k from slope
                    slope_1st, intercept_1st = np.polyfit(time_lin_1st, log_term_1st, 1)
                    k_linear_1st = -slope_1st
                    r2_linear_1st = r2_score(
                        log_term_1st, slope_1st * time_lin_1st + intercept_1st
                    )

                    ax2.scatter(
                        time_lin_1st,
                        log_term_1st,
                        color=color,
                        s=20,
                        alpha=0.6,
                        label=f"{label}",
                    )
                    ax2.plot(
                        time_lin_1st,
                        slope_1st * time_lin_1st + intercept_1st,
                        color=color,
                        linewidth=2,
                        linestyle="--",
                        label=f"{label} fit (k={k_linear_1st:.2e}, R²={r2_linear_1st:.4f})",
                    )

            except Exception as e:
                print(f"Warning: 1st order fitting failed for {label}: {str(e)}")
                popt_1st = None
                r2_1st = None
                k_linear_1st = None
                r2_linear_1st = None

            # ============ SECOND ORDER FITTING ============
            try:
                popt_2nd, pcov_2nd = curve_fit(
                    second_order_model,
                    time_fit,
                    mass_fit,
                    p0=[m_0_guess, k_guess_second],
                    maxfev=10000,
                    bounds=([95, 0], [100, 0.1]),
                )
                m_0_2nd, k_2nd = popt_2nd
                mass_pred_2nd = second_order_model(time_fit, *popt_2nd)
                r2_2nd = r2_score(mass_fit, mass_pred_2nd)
                perr_2nd = np.sqrt(np.diag(pcov_2nd))

                # Plot 2nd order fit on main plot
                mass_smooth_2nd = second_order_model(time_smooth, *popt_2nd)
                ax1.plot(
                    time_smooth,
                    mass_smooth_2nd,
                    label=f"{label} (2nd order, R²={r2_2nd:.4f})",
                    color=color,
                    linewidth=1.5,
                    linestyle=":",
                )

                # Linearized 2nd order plot: 1/m vs t
                valid_mask_2nd = mass_fit > 0.01
                if np.sum(valid_mask_2nd) > 5:
                    time_lin_2nd = time_fit[valid_mask_2nd]
                    inv_term_2nd = 1.0 / mass_fit[valid_mask_2nd]

                    # Linear fit to get k from slope
                    slope_2nd, intercept_2nd = np.polyfit(time_lin_2nd, inv_term_2nd, 1)
                    k_linear_2nd = slope_2nd / m_0_2nd
                    r2_linear_2nd = r2_score(
                        inv_term_2nd, slope_2nd * time_lin_2nd + intercept_2nd
                    )

                    ax3.scatter(
                        time_lin_2nd,
                        inv_term_2nd,
                        color=color,
                        s=20,
                        alpha=0.6,
                        label=f"{label}",
                    )
                    ax3.plot(
                        time_lin_2nd,
                        slope_2nd * time_lin_2nd + intercept_2nd,
                        color=color,
                        linewidth=2,
                        linestyle="--",
                        label=f"{label} fit (k={k_linear_2nd:.2e}, R²={r2_linear_2nd:.4f})",
                    )

            except Exception as e:
                print(f"Warning: 2nd order fitting failed for {label}: {str(e)}")
                popt_2nd = None
                r2_2nd = None
                k_linear_2nd = None
                r2_linear_2nd = None

            # Plot experimental data
            ax1.plot(
                time_data,
                mass_data,
                color=color,
                linewidth=1,
                alpha=0.7,
                label=f"{label} (data)",
            )

            # Print results
            print(f"\n{'='*60}")
            print(f"{label}")
            print(f"{'='*60}")

            if popt_1st is not None:
                print(f"\n1st Order Model (Non-linear fit):")
                print(f"  Rate constant (k): {k_1st:.6e} ± {perr_1st[1]:.6e} min^-1")
                print(f"  Initial mass (m_0): {m_0_1st:.2f} ± {perr_1st[0]:.2f}%")
                print(f"  Final mass (m_∞): 0.00% (complete degradation)")
                print(f"  R² (non-linear): {r2_1st:.6f}")
                if k_linear_1st is not None:
                    print(f"  k from linearized plot: {k_linear_1st:.6e} min^-1")
                    print(f"  R² (linearized): {r2_linear_1st:.6f}")
                print(f"  Half-life: {np.log(2)/k_1st:.2f} min")

            if popt_2nd is not None:
                print(f"\n2nd Order Model (Non-linear fit):")
                print(
                    f"  Rate constant (k): {k_2nd:.6e} ± {perr_2nd[1]:.6e} %^-1·min^-1"
                )
                print(f"  Initial mass (m_0): {m_0_2nd:.2f} ± {perr_2nd[0]:.2f}%")
                print(f"  Final mass (m_∞): 0.00% (complete degradation)")
                print(f"  R² (non-linear): {r2_2nd:.6f}")
                if k_linear_2nd is not None:
                    print(f"  k from linearized plot: {k_linear_2nd:.6e} %^-1·min^-1")
                    print(f"  R² (linearized): {r2_linear_2nd:.6f}")

            if popt_1st is not None and popt_2nd is not None:
                print(f"\nModel Comparison:")
                print(
                    f"  Better fit: {'1st order' if r2_1st > r2_2nd else '2nd order'}"
                )
                print(f"  ΔR²: {abs(r2_1st - r2_2nd):.6f}")

            # Store results
            results.append(
                {
                    "label": label,
                    "first_order": (
                        {
                            "k": k_1st,
                            "r2": r2_1st,
                            "k_linear": k_linear_1st,
                            "r2_linear": r2_linear_1st,
                        }
                        if popt_1st is not None
                        else None
                    ),
                    "second_order": (
                        {
                            "k": k_2nd,
                            "r2": r2_2nd,
                            "k_linear": k_linear_2nd,
                            "r2_linear": r2_linear_2nd,
                        }
                        if popt_2nd is not None
                        else None
                    ),
                }
            )

        # Add reference line
        ax1.axhline(y=100, color="r", linestyle="--", linewidth=0.5, label="100% Mass")

        # Add legends
        ax1.legend(fontsize=7, loc="best")
        ax2.legend(fontsize=7, loc="best")
        ax3.legend(fontsize=7, loc="best")

        plt.tight_layout()

        # Save plots
        plt.savefig(
            self.result_dir
            / f"{self.result_name}tga_isothermal_{isothermal_temp}_{target_mass}m_z_kinetic_analysis.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir
            / f"{self.result_name}tga_isothermal_{isothermal_temp}_{target_mass}m_z_kinetic_analysis.eps",
            format="eps",
            dpi=400,
        )

        return results

    def calculate_baselines(
        self,
        tga_data: pd.DataFrame,
        top_baseline_range: tuple = None,
        bottom_baseline_range: tuple = None,
        temp_or_time: str = "Temp",
    ):
        """
        Calculate top and bottom baselines for TGA data.

        Args:
            tga_data: DataFrame with TGA data
            top_baseline_range: Tuple (start, end) for top baseline region (temp or time)
            bottom_baseline_range: Tuple (start, end) for bottom baseline region (temp or time)
            temp_or_time: "Temp" or "Time" to specify which column to use

        Returns:
            tuple: (top_baseline, bottom_baseline) as average mass % in each region
        """
        x_col = "Temp" if temp_or_time == "Temp" else "Time/min"

        # Calculate top baseline (initial stable region)
        if top_baseline_range is not None:
            top_mask = (tga_data[x_col] >= top_baseline_range[0]) & (
                tga_data[x_col] <= top_baseline_range[1]
            )
            top_baseline = tga_data.loc[top_mask, "Mass loss/pct"].mean()
        else:
            # Use first 10% of data as top baseline
            n_points = max(5, len(tga_data) // 10)
            top_baseline = tga_data["Mass loss/pct"].iloc[:n_points].mean()

        # Calculate bottom baseline (final stable region)
        if bottom_baseline_range is not None:
            bottom_mask = (tga_data[x_col] >= bottom_baseline_range[0]) & (
                tga_data[x_col] <= bottom_baseline_range[1]
            )
            bottom_baseline = tga_data.loc[bottom_mask, "Mass loss/pct"].mean()
        else:
            # Use last 10% of data as bottom baseline
            n_points = max(5, len(tga_data) // 10)
            bottom_baseline = tga_data["Mass loss/pct"].iloc[-n_points:].mean()

        return top_baseline, bottom_baseline

    def mass_to_conversion(
        self, mass_pct: float, top_baseline: float, bottom_baseline: float
    ):
        """
        Convert mass percentage to conversion percentage using baselines.

        Args:
            mass_pct: Current mass percentage
            top_baseline: Initial stable mass percentage (0% conversion)
            bottom_baseline: Final stable mass percentage (100% conversion)

        Returns:
            float: Conversion percentage (0-100%)
        """
        if abs(top_baseline - bottom_baseline) < 0.01:
            # Avoid division by zero
            return 0.0

        conversion = (top_baseline - mass_pct) / (top_baseline - bottom_baseline) * 100
        # Clip to 0-100% range
        return np.clip(conversion, 0, 100)

    def conversion_to_mass(
        self, conversion_pct: float, top_baseline: float, bottom_baseline: float
    ):
        """
        Convert conversion percentage to mass percentage using baselines.

        Args:
            conversion_pct: Conversion percentage (0-100%)
            top_baseline: Initial stable mass percentage (0% conversion)
            bottom_baseline: Final stable mass percentage (100% conversion)

        Returns:
            float: Mass percentage
        """
        mass_pct = top_baseline - (conversion_pct / 100) * (
            top_baseline - bottom_baseline
        )
        return mass_pct

    def plot_flynn_ozawa_wall_astm(
        self,
        heating_rates: list[float],
        conversion_levels: list[float],
        xlim: tuple = (250, 400),
        ylim: tuple = (1, 4),
        ea_ylim: tuple = (0, 300),
        initial_correction_temp: int = 230,
        max_iterations: int = 10,
        tolerance: float = 0.01,
        top_baseline_range: tuple = None,
        bottom_baseline_range: tuple = None,
        heating_rate_colors: list[str] = None,
    ):
        """
        Plot Flynn-Ozawa-Wall analysis using ASTM E1641-23 standard.

        This method uses numerical integration constants from ASTM E1641-23 Table 1
        for more accurate activation energy determination. Conversion is calculated
        using actual baselines rather than assuming 100% to 0% range.

        Args:
            heating_rates: List of heating rates (°C/min)
            conversion_levels: List of conversion levels in % (e.g., [10, 20, 30, ...])
            xlim: X-axis limits (temperature range in °C)
            ylim: Y-axis limits (log(β) range)
            ea_ylim: Y-axis limits for Ea plot (kJ/mol)
            initial_correction_temp: Temperature correction
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance for Ea (kJ/mol)
            top_baseline_range: Tuple (start_temp, end_temp) for top baseline region.
                              If None, uses first 10% of data.
            bottom_baseline_range: Tuple (start_temp, end_temp) for bottom baseline region.
                                 If None, uses last 10% of data.
            heating_rate_colors: List of colors for heating rate data points (one per rate).
                               If None, uses rocket colormap.
                               Fitted lines always use rocket colormap.

        Returns:
            dict: Dictionary containing temperatures, heating rates, and Ea values
        """

        # ASTM E1641-23 Table 1: Numerical Integration Constants
        astm_table = {
            8: (5.3699, 0.5398),
            9: (5.8980, 0.5281),
            10: (6.4167, 0.5187),
            11: (6.928, 0.511),
            12: (7.433, 0.505),
            13: (7.933, 0.500),
            14: (8.427, 0.494),
            15: (8.918, 0.491),
            16: (9.406, 0.488),
            17: (9.890, 0.484),
            18: (10.372, 0.482),
            19: (10.851, 0.479),
            20: (11.3277, 0.4770),
            21: (11.803, 0.475),
            22: (12.276, 0.473),
            23: (12.747, 0.471),
            24: (13.217, 0.470),
            25: (13.686, 0.469),
            26: (14.153, 0.467),
            27: (14.619, 0.466),
            28: (15.084, 0.465),
            29: (15.547, 0.463),
            30: (16.0104, 0.4629),
            31: (16.472, 0.462),
            32: (16.933, 0.461),
            33: (17.394, 0.461),
            34: (17.853, 0.459),
            35: (18.312, 0.459),
            36: (18.770, 0.458),
            37: (19.228, 0.458),
            38: (19.684, 0.456),
            39: (20.141, 0.456),
            40: (20.5967, 0.4558),
            41: (21.052, 0.455),
            42: (21.507, 0.455),
            43: (21.961, 0.454),
            44: (22.415, 0.454),
            45: (22.868, 0.453),
            46: (23.321, 0.453),
            47: (23.774, 0.453),
            48: (24.226, 0.452),
            49: (24.678, 0.452),
            50: (25.1295, 0.4515),
            51: (25.5806, 0.4511),
            52: (26.0314, 0.4508),
            53: (26.4820, 0.4506),
            54: (26.9323, 0.4503),
            55: (27.3823, 0.4500),
            56: (27.8319, 0.4498),
            57: (28.2814, 0.4495),
            58: (28.7305, 0.4491),
            59: (29.1794, 0.4489),
            60: (29.6281, 0.4487),
        }

        def get_astm_constants(e_over_rt):
            """Interpolate a and b values from ASTM table."""
            e_rt_values = np.array(list(astm_table.keys()))

            if e_over_rt < e_rt_values[0]:
                return astm_table[e_rt_values[0]]
            elif e_over_rt > e_rt_values[-1]:
                return astm_table[e_rt_values[-1]]
            else:
                # Linear interpolation
                idx = np.searchsorted(e_rt_values, e_over_rt)
                if idx == 0:
                    return astm_table[e_rt_values[0]]

                e_rt_low = e_rt_values[idx - 1]
                e_rt_high = e_rt_values[idx]

                a_low, b_low = astm_table[e_rt_low]
                a_high, b_high = astm_table[e_rt_high]

                # Interpolate
                frac = (e_over_rt - e_rt_low) / (e_rt_high - e_rt_low)
                a = a_low + frac * (a_high - a_low)
                b = b_low + frac * (b_high - b_low)

                return a, b

        if len(heating_rates) != len(self.tga_data_path):
            raise ValueError(
                f"Number of heating rates must match number of TGA data paths"
            )

        R = 8.314  # Gas constant in J/(mol·K)

        # Create two separate figures
        fig1, ax1 = plt.subplots(1, figsize=(6, 4.5))
        fig2, ax2 = plt.subplots(1, figsize=(6, 4.5))

        # Aesthetics
        ax1.set_xlabel("1000/T (K$^{-1}$)", fontsize=10)
        ax1.set_ylabel("log(β) [log(K/min)]", fontsize=10)
        ax1.spines["top"].set_visible(False)
        ax1.tick_params(axis="y", which="major", labelsize=8, direction="in")
        ax1.tick_params(axis="x", which="major", labelsize=8, direction="in")

        # Add secondary x-axis on top for Temperature (°C)
        ax1_top = ax1.secondary_xaxis(
            "top",
            functions=(lambda x: 1000.0 / x - 273.15, lambda x: 1000.0 / (x + 273.15)),
        )
        ax1_top.set_xlabel("Temperature (°C)", fontsize=10)
        ax1_top.tick_params(axis="x", which="major", labelsize=8, direction="in")

        # Add secondary y-axis on right for actual heating rate β (°C/min)
        # Use log10 since the main axis uses log10(β)
        ax1_right = ax1.secondary_yaxis(
            "right",
            functions=(lambda y: 10**y, lambda y: np.log10(y)),
        )
        ax1_right.set_ylabel("β (K/min)", fontsize=10)
        ax1_right.tick_params(axis="y", which="major", labelsize=8, direction="in")

        # Set specific tick marks for common heating rates
        common_heating_rates = [1, 2, 5, 10, 20, 30, 40, 50, 100]
        ax1_right.set_yticks(common_heating_rates)

        xlim_inv_T = 1000.0 / (np.array(xlim) + 273.15)
        # ax1.set_xlim(xlim_inv_T)
        # ax1.set_ylim(ylim)

        ax2.set_xlabel("Conversion (%)", fontsize=10)
        ax2.set_ylabel("Apparent Activation Energy (kJ/mol)", fontsize=10)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.tick_params(axis="both", which="major", labelsize=8, direction="in")
        # ax2.set_xlim(40, 100)
        # ax2.set_ylim(ea_ylim)

        results = {}
        ea_values = []
        conv_values = []

        # Create rocket colormap for fitted lines
        # Try to import rocket from seaborn, fall back to inferno if not available
        try:
            import seaborn as sns

            rocket_cmap = sns.color_palette("rocket", as_cmap=True)
        except ImportError:
            from matplotlib import cm

            rocket_cmap = cm.get_cmap("inferno")

        n_conversions = len(conversion_levels)
        line_colors = [
            rocket_cmap(i / (n_conversions - 1)) for i in range(n_conversions)
        ]

        # Create or use provided heating rate colors for data points
        if heating_rate_colors is None:
            # Use rocket colormap for heating rate colors as well
            n_rates = len(heating_rates)
            heating_rate_colors = [
                rocket_cmap(i / (n_rates - 1)) for i in range(n_rates)
            ]
        elif len(heating_rate_colors) != len(heating_rates):
            raise ValueError(
                f"Number of heating_rate_colors ({len(heating_rate_colors)}) "
                f"must match number of heating_rates ({len(heating_rates)})"
            )

        print("\n" + "=" * 80)
        print("ASTM E1641-23 Flynn-Ozawa-Wall Analysis")
        print("=" * 80)

        # First pass: calculate baselines for all datasets
        baselines_dict = {}
        for tga_path, label in zip(self.tga_data_path, self.labels):
            tga_data = pd.read_csv(
                self.data_dir / tga_path,
                encoding="iso-8859-1",
                on_bad_lines="skip",
                skiprows=37,
            )
            tga_data, _ = self.preprocess(
                tga_data, None, initial_correction_temp, "Temp"
            )

            top_baseline, bottom_baseline = self.calculate_baselines(
                tga_data, top_baseline_range, bottom_baseline_range, "Temp"
            )
            baselines_dict[label] = (top_baseline, bottom_baseline)

            print(f"\n{label}:")
            print(f"  Top baseline: {top_baseline:.2f}%")
            print(f"  Bottom baseline: {bottom_baseline:.2f}%")
            print(f"  Total mass loss: {top_baseline - bottom_baseline:.2f}%")

        for conv_idx, conversion in enumerate(conversion_levels):
            temperatures = []
            beta_values = []

            for tga_path, heating_rate, label in zip(
                self.tga_data_path, heating_rates, self.labels
            ):
                tga_data = pd.read_csv(
                    self.data_dir / tga_path,
                    encoding="iso-8859-1",
                    on_bad_lines="skip",
                    skiprows=37,
                )
                tga_data, _ = self.preprocess(
                    tga_data, None, initial_correction_temp, "Temp"
                )

                # Get baselines for this dataset
                top_baseline, bottom_baseline = baselines_dict[label]

                # Convert conversion % to target mass % using baselines
                target_mass = self.conversion_to_mass(
                    conversion, top_baseline, bottom_baseline
                )

                idx = (tga_data["Mass loss/pct"] - target_mass).abs().idxmin()
                temp_at_conversion = tga_data.loc[idx, "Temp"]

                temperatures.append(temp_at_conversion)
                beta_values.append(heating_rate)

            temperatures = np.array(temperatures)
            beta_values = np.array(beta_values)
            T_kelvin = temperatures + 273.15
            inv_T = 1000.0 / T_kelvin
            log_beta = np.log10(beta_values)

            # Iterative method using ASTM constants
            # Using Ozawa-Flynn-Wall equation: log(β) = const - 1.052*E/(RT)
            # Perform linear regression once (data doesn't change)
            slope, intercept, r_value, _, _ = linregress(inv_T, log_beta)
            r2 = r_value**2

            # Initial guess using standard Ozawa constant
            Ea = -slope * R / 0.457  # Initial guess in kJ/mol using OFW constant

            for iteration in range(max_iterations):
                Ea_old = Ea
                T_mean = np.mean(T_kelvin)
                e_over_rt = (Ea * 1000) / (R * T_mean)
                a, b = get_astm_constants(e_over_rt)

                # Recalculate Ea using updated b constant
                Ea = -slope * R / b  # in kJ/mol

                if abs(Ea - Ea_old) < tolerance:
                    break

            results[conversion] = {
                "temperatures": temperatures,
                "heating_rates": beta_values,
                "inv_T": inv_T,
                "log_beta": log_beta,
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "Ea": Ea,
                "a": a,
                "b": b,  # Original ASTM b (for log₁₀)
                "e_over_rt": e_over_rt,
            }

            ea_values.append(Ea)
            conv_values.append(conversion)

            # Plot data points - each point gets the color of its heating rate
            for i in range(len(inv_T)):
                ax1.plot(
                    inv_T[i],
                    log_beta[i],
                    marker="o",
                    markersize=6,
                    color=heating_rate_colors[i],
                    zorder=3,  # Ensure points are on top
                )

            # Plot fitted line extending slightly beyond data points
            inv_T_range = inv_T.max() - inv_T.min()
            inv_T_fit = np.linspace(
                inv_T.min() - 0.02 * inv_T_range,
                inv_T.max() + 0.02 * inv_T_range,
                100,
            )
            log_beta_fit = slope * inv_T_fit + intercept
            ax1.plot(
                inv_T_fit,
                log_beta_fit,
                linestyle="--",
                linewidth=1.5,
                color=line_colors[conv_idx],
                alpha=0.7,
                zorder=2,  # Lines behind points
                label=f"{conversion:.0f}% (Ea={Ea:.1f} kJ/mol, R²={r2:.3f})",
            )

            print(f"\nConversion: {conversion:.2f}%")
            print(f"  Temperatures: {temperatures} °C")
            print(f"  Heating rates: {beta_values} K/min")
            print(f"  1000/T values: {inv_T}")
            print(f"  log10(β) values: {log_beta}")
            print(f"  Regression: slope={slope:.4f}, intercept={intercept:.4f}")
            print(f"  E/RT: {e_over_rt:.2f}, ASTM b={b:.4f}")
            print(f"  Ea (ASTM): {Ea:.2f} kJ/mol, R²={r2:.4f}")

        # Plot Ea vs conversion
        ax2.plot(
            conv_values,
            ea_values,
            marker="o",
            markersize=8,
            linewidth=2,
            color="#2E86AB",
            label="Ea (ASTM E1641)",
        )

        avg_Ea = np.mean(ea_values)
        std_Ea = np.std(ea_values)
        ax2.axhline(
            y=avg_Ea,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Avg = {avg_Ea:.1f} ± {std_Ea:.1f} kJ/mol",
        )

        ax1.legend(fontsize=8, loc="best")
        # ax1.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        ax2.legend(fontsize=9, loc="best")
        # ax2.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

        # Save figures
        fig1.tight_layout()
        fig1.savefig(
            self.result_dir / f"{self.result_name}fow_astm_logbeta_vs_invT.png",
            dpi=400,
        )
        fig1.savefig(
            self.result_dir / f"{self.result_name}fow_astm_logbeta_vs_invT.eps",
            format="eps",
            dpi=400,
        )

        fig2.tight_layout()
        fig2.savefig(
            self.result_dir / f"{self.result_name}fow_astm_Ea_vs_conversion.png",
            dpi=400,
        )
        fig2.savefig(
            self.result_dir / f"{self.result_name}fow_astm_Ea_vs_conversion.eps",
            format="eps",
            dpi=400,
        )

        # Export CSV with label
        export_data = pd.DataFrame(
            {
                "experiment_label": [self.result_name.rstrip("_")] * len(conv_values),
                "conversion_%": conv_values,
                "activation_energy_kJ_mol": ea_values,
                "r2": [results[conv]["r2"] for conv in conv_values],
                "E_over_RT": [results[conv]["e_over_rt"] for conv in conv_values],
                "ASTM_b": [results[conv]["b"] for conv in conv_values],
            }
        )

        csv_filename = f"{self.result_name}fow_astm_data.csv"
        export_data.to_csv(self.result_dir / csv_filename, index=False)

        # Also store baselines in results for reference
        results["baselines"] = baselines_dict

        print("\n" + "=" * 80)
        print(f"Avg Ea (ASTM): {avg_Ea:.2f} ± {std_Ea:.2f} kJ/mol")
        print(f"Range: {min(ea_values):.2f} - {max(ea_values):.2f} kJ/mol")
        print("=" * 80)
        print(f"\nASTM E1641-23 plots saved:")
        print(f"  - {self.result_name}fow_astm_logbeta_vs_invT.png/eps")
        print(f"  - {self.result_name}fow_astm_Ea_vs_conversion.png/eps")
        print(f"  - {csv_filename}")

        plt.close(fig1)
        plt.close(fig2)

        return results

    def compare_flynn_ozawa_wall_experiments(
        self,
        csv_file_paths: list[Path],
        labels: list[str] = None,
        colors: list[str] = None,
        ylim: tuple = (0, 300),
        xlim: tuple = (0, 100),
    ):
        """
        Compare activation energy vs conversion from multiple Flynn-Ozawa-Wall experiments.

        Args:
            csv_file_paths: List of paths to FOW CSV files (can be relative to result_dir)
            labels: List of labels for each experiment. If None, uses experiment_label from CSV
            colors: List of colors for each experiment. If None, uses default color scheme
            ylim: Y-axis limits for Ea plot (kJ/mol)
            xlim: X-axis limits for conversion (%)

        Returns:
            pd.DataFrame: Combined data from all experiments
        """

        # Read all CSV files
        all_data = []
        for csv_path in csv_file_paths:
            # Try to read from result_dir first, then as absolute path
            if not csv_path.is_absolute():
                csv_path = self.result_dir / csv_path

            df = pd.read_csv(csv_path)
            all_data.append(df)

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)

        # Get unique experiments
        if "experiment_label" in combined_data.columns:
            experiments = combined_data["experiment_label"].unique()
        else:
            experiments = [f"Experiment {i+1}" for i in range(len(all_data))]

        # Use provided labels or default to experiment names
        if labels is None:
            labels = experiments
        elif len(labels) != len(experiments):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match number of experiments ({len(experiments)})"
            )

        # Use provided colors or default color scheme
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
        elif len(colors) != len(experiments):
            raise ValueError(
                f"Number of colors ({len(colors)}) must match number of experiments ({len(experiments)})"
            )

        # Create comparison plot
        fig, ax = plt.subplots(1, figsize=(4, 6))
        plt.subplots_adjust(hspace=0.5)

        # Aesthetics
        ax.set_xlabel("Conversion (%)", fontsize=10)
        ax.set_ylabel("Apparent Activation Energy (kJ/mol)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8, direction="in")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # First pass: collect data and determine which curves are top/bottom
        experiment_data = []
        for i, (exp, label, color) in enumerate(zip(experiments, labels, colors)):
            if "experiment_label" in combined_data.columns:
                exp_data = combined_data[combined_data["experiment_label"] == exp]
            else:
                exp_data = all_data[i]

            conv = exp_data["conversion_%"]
            ea = exp_data["activation_energy_kJ_mol"]
            avg_ea = ea.mean()
            std_ea = ea.std()

            experiment_data.append(
                {
                    "exp": exp,
                    "label": label,
                    "color": color,
                    "conv": conv,
                    "ea": ea,
                    "avg_ea": avg_ea,
                    "std_ea": std_ea,
                }
            )

        # Sort by average Ea to determine top and bottom curves
        experiment_data_sorted = sorted(
            experiment_data, key=lambda x: x["avg_ea"], reverse=True
        )

        # Plot each experiment with smart label positioning
        for idx, exp_info in enumerate(experiment_data_sorted):
            conv = exp_info["conv"]
            ea = exp_info["ea"]
            label = exp_info["label"]
            color = exp_info["color"]
            avg_ea = exp_info["avg_ea"]
            std_ea = exp_info["std_ea"]

            # Plot data points
            ax.plot(
                conv,
                ea,
                marker="o",
                markersize=6,
                linewidth=1.5,
                color=color,
                label=f"{label} (Avg={avg_ea:.1f}±{std_ea:.1f} kJ/mol)",
                alpha=0.8,
            )

            # Determine label position based on curve position
            # Top curve (highest Ea) -> labels above
            # Bottom curve (lowest Ea) -> labels below
            if idx == 0:  # Top curve
                xytext = (0, 10)
                va = "bottom"
            else:  # Bottom curve(s)
                xytext = (0, -10)
                va = "top"

            # Add labels to each data point with activation energy
            for j, (c, e) in enumerate(zip(conv, ea)):
                ax.annotate(
                    f"{e:.1f}",
                    (c, e),
                    textcoords="offset points",
                    xytext=xytext,
                    ha="center",
                    va=va,
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                )

            # Plot average line
            ax.axhline(
                y=avg_ea,
                color=color,
                linestyle="--",
                linewidth=1,
                alpha=0.4,
            )

            print(f"\n{label}:")
            print(f"  Average Ea: {avg_ea:.2f} ± {std_ea:.2f} kJ/mol")
            print(f"  Range: {ea.min():.2f} - {ea.max():.2f} kJ/mol")
            print(f"  Number of points: {len(ea)}")

        ax.legend(fontsize=9, loc="best")
        # ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

        # Set x-axis ticks to match actual conversion levels
        unique_conversions = sorted(combined_data["conversion_%"].unique())
        ax.set_xticks(unique_conversions)

        plt.tight_layout()

        # Save comparison plot
        comparison_filename = "fow_comparison"
        plt.savefig(
            self.result_dir / f"{comparison_filename}_Ea_vs_conversion.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir / f"{comparison_filename}_Ea_vs_conversion.eps",
            format="eps",
            dpi=400,
        )

        # Export combined data
        combined_csv_filename = f"{comparison_filename}_combined_data.csv"
        combined_data.to_csv(self.result_dir / combined_csv_filename, index=False)

        print(f"\n{'='*80}")
        print("Comparison plots saved:")
        print(f"  - {comparison_filename}_Ea_vs_conversion.png/eps")
        print(f"  - {combined_csv_filename}")
        print(f"{'='*80}")

        plt.close(fig)

        return combined_data

    def compare_isothermal_degradation_rates(
        self,
        mass_loss_targets: list[float] = [10, 20, 30, 40, 50, 60, 70, 80],
        initial_correction_time: int = 50,
        xlim: tuple = (0, 1200),
        ylim: tuple = (0, 100),
    ):
        """
        Compare isothermal TGA curves showing the time each sample takes to reach
        specific mass loss percentages.

        Args:
            mass_loss_targets: List of target mass loss % to compare (e.g., [10, 20, 50])
            initial_correction_time: Time correction for initial period
            xlim: X-axis limits for plotting (time in minutes)
            ylim: Y-axis limits for plotting (mass %)

        Returns:
            pd.DataFrame: Table with time to reach each mass loss % for all samples
        """

        # Process all datasets
        all_data = []
        for tga_path, label, color in zip(self.tga_data_path, self.labels, self.colors):
            # Read TGA data
            tga_data = pd.read_csv(
                self.data_dir / tga_path, encoding="iso-8859-1", on_bad_lines="skip"
            )
            num_skiprows = _find_empty_row(tga_data)
            tga_data = pd.read_csv(
                self.data_dir / tga_path,
                encoding="iso-8859-1",
                on_bad_lines="skip",
                skiprows=num_skiprows,
            )

            # Preprocess data
            tga_data, _ = self.preprocess(
                tga_data, None, initial_correction_time, "Time"
            )

            # Store processed data
            all_data.append(
                {
                    "label": label,
                    "color": color,
                    "data": tga_data,
                    "time": tga_data["Time/min"].values - initial_correction_time,
                    "mass": tga_data["Mass loss/pct"].values,
                }
            )

        # Create results table
        results = []

        print("\n" + "=" * 80)
        print(f"Isothermal Degradation Rate Comparison")
        print("=" * 80)

        # For each mass loss target
        for target_mass in mass_loss_targets:
            row_data = {"mass_pct": 100 - target_mass, "mass_loss_pct": target_mass}
            target_mass_value = 100 - target_mass

            # Find time for each sample to reach this mass loss
            sample_times = {}
            for sample in all_data:
                if sample["mass"].min() <= target_mass_value <= sample["mass"].max():
                    idx = np.abs(sample["mass"] - target_mass_value).argmin()
                    sample_time = sample["time"][idx]
                    row_data[f"{sample['label']}_time_min"] = sample_time
                    sample_times[sample["label"]] = sample_time
                else:
                    row_data[f"{sample['label']}_time_min"] = np.nan
                    sample_times[sample["label"]] = np.nan

            # Calculate rate improvement factors (pairwise comparisons)
            # For each pair of samples, calculate how much faster one is than the other
            for i, sample_i in enumerate(all_data):
                for j, sample_j in enumerate(all_data):
                    if i < j:  # Only calculate once for each pair
                        time_i = sample_times.get(sample_i["label"], np.nan)
                        time_j = sample_times.get(sample_j["label"], np.nan)

                        if not np.isnan(time_i) and not np.isnan(time_j) and time_j > 0:
                            # Rate improvement factor: how much faster is sample_j compared to sample_i
                            rate_factor = time_i / time_j
                            row_data[
                                f"{sample_j['label']}_vs_{sample_i['label']}_rate_factor"
                            ] = rate_factor

            results.append(row_data)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Print results table
        print("\n" + "-" * 80)
        print(f"Time (min) to reach each mass loss percentage:")
        print("-" * 80)
        print(f"{'Mass %':<12} {'Loss %':<12}", end="")
        for sample in all_data:
            print(f"{sample['label'][:18]:<20}", end="")
        print()
        print("-" * 80)

        for _, row in results_df.iterrows():
            mass_pct = row["mass_pct"]
            mass_loss_pct = row["mass_loss_pct"]
            print(f"{mass_pct:<12.1f} {mass_loss_pct:<12.1f}", end="")

            for sample in all_data:
                time_col = f"{sample['label']}_time_min"
                if time_col in row and not pd.isna(row[time_col]):
                    print(f"{row[time_col]:<20.1f}", end="")
                else:
                    print(f"{'N/A':<20}", end="")
            print()

        print("-" * 80)

        # Print rate improvement factors
        if len(all_data) == 2:
            print(f"\nRate Improvement Factors:")
            print("-" * 80)
            sample_0_label = all_data[0]["label"]
            sample_1_label = all_data[1]["label"]
            rate_col = f"{sample_1_label}_vs_{sample_0_label}_rate_factor"

            if rate_col in results_df.columns:
                print(
                    f"{'Mass %':<12} {'Loss %':<12} {sample_1_label} vs {sample_0_label}"
                )
                print("-" * 80)
                for _, row in results_df.iterrows():
                    if not pd.isna(row[rate_col]):
                        mass_pct = row["mass_pct"]
                        mass_loss_pct = row["mass_loss_pct"]
                        rate = row[rate_col]
                        if rate > 1:
                            print(
                                f"{mass_pct:<12.1f} {mass_loss_pct:<12.1f} {rate:.2f}× faster"
                            )
                        else:
                            print(
                                f"{mass_pct:<12.1f} {mass_loss_pct:<12.1f} {1/rate:.2f}× slower"
                            )
                print("-" * 80)

        # Create visualization - single plot with TGA curves
        fig, ax1 = plt.subplots(1, figsize=(6, 4))
        plt.subplots_adjust(hspace=0.5)

        # Plot TGA curves with markers at comparison points
        for sample in all_data:
            # Only plot data within xlim range to avoid artifacts
            time_mask = (sample["time"] >= xlim[0]) & (sample["time"] <= xlim[1])
            ax1.plot(
                sample["time"][time_mask],
                sample["mass"][time_mask],
                label=sample["label"],
                color=sample["color"],
                linewidth=1,
            )

            # Add markers at each mass loss target (only within xlim range)
            for target_mass in mass_loss_targets:
                target_mass_value = 100 - target_mass
                if sample["mass"].min() <= target_mass_value <= sample["mass"].max():
                    idx = np.abs(sample["mass"] - target_mass_value).argmin()
                    marker_time = sample["time"][idx]
                    marker_mass = sample["mass"][idx]

                    # Only add marker if it's within xlim range
                    if xlim[0] <= marker_time <= xlim[1]:
                        ax1.scatter(
                            marker_time,
                            marker_mass,
                            color=sample["color"],
                            s=40,
                            zorder=5,
                        )

        # Styling - match plot_tga_isothermal
        ax1.set_xlabel("Time (min)", fontsize=10)
        ax1.set_ylabel("Mass (%)", fontsize=10)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.tick_params(axis="both", which="major", labelsize=8, direction="in")
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_yticks([0, 20, 40, 60, 80, 100])
        ax1.legend(fontsize=10)
        # ax1.grid(True, linestyle="-", alpha=0.2, linewidth=0.5, color="gray")

        plt.tight_layout()

        # Save plots
        plt.savefig(
            self.result_dir / f"{self.result_name}isothermal_rate_comparison.png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.savefig(
            self.result_dir / f"{self.result_name}isothermal_rate_comparison.eps",
            format="eps",
            dpi=400,
            bbox_inches="tight",
        )

        # Save results to CSV
        csv_filename = f"{self.result_name}isothermal_rate_comparison.csv"
        results_df.to_csv(self.result_dir / csv_filename, index=False)

        print("\n" + "=" * 80)
        print("Comparison complete!")
        print(f"Plots saved: {self.result_name}isothermal_rate_comparison.png/eps")
        print(f"Data saved: {csv_filename}")
        print("=" * 80)

        plt.show()

        return results_df
