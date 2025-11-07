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

        ax.legend(fontsize=7)
        ax.set_xlim(xlim)
        ax.set_yticks([0, 20, 40, 60, 80, 100])  # Add this line
        ax.set_ylim(0, 105)
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
        # Use data points before 2% mass loss for baseline
        baseline_data = tga_data[tga_data["Mass loss/pct"] < 2.0]
        if len(baseline_data) < 2:
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
        2. Linearized 1st order plot: ln(m) vs time
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
        ax2.set_ylabel("ln(m)", fontsize=10)
        ax2.set_title("1st Order: ln(m) vs t", fontsize=10)

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

                # Linearized 1st order plot: ln(m) vs t
                valid_mask_1st = mass_fit > 0.01
                if np.sum(valid_mask_1st) > 5:
                    time_lin_1st = time_fit[valid_mask_1st]
                    ln_term_1st = np.log(mass_fit[valid_mask_1st])

                    # Linear fit to get k from slope
                    slope_1st, intercept_1st = np.polyfit(time_lin_1st, ln_term_1st, 1)
                    k_linear_1st = -slope_1st
                    r2_linear_1st = r2_score(
                        ln_term_1st, slope_1st * time_lin_1st + intercept_1st
                    )

                    ax2.scatter(
                        time_lin_1st,
                        ln_term_1st,
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
                print(f"  Rate constant (k): {k_2nd:.6e} ± {perr_2nd[1]:.6e} %^-1·min^-1")
                print(f"  Initial mass (m_0): {m_0_2nd:.2f} ± {perr_2nd[0]:.2f}%")
                print(f"  Final mass (m_∞): 0.00% (complete degradation)")
                print(f"  R² (non-linear): {r2_2nd:.6f}")
                if k_linear_2nd is not None:
                    print(f"  k from linearized plot: {k_linear_2nd:.6e} %^-1·min^-1")
                    print(f"  R² (linearized): {r2_linear_2nd:.6f}")

            if popt_1st is not None and popt_2nd is not None:
                print(f"\nModel Comparison:")
                print(f"  Better fit: {'1st order' if r2_1st > r2_2nd else '2nd order'}")
                print(f"  ΔR²: {abs(r2_1st - r2_2nd):.6f}")

            # Store results
            results.append({
                "label": label,
                "first_order": {"k": k_1st, "r2": r2_1st, "k_linear": k_linear_1st, "r2_linear": r2_linear_1st} if popt_1st is not None else None,
                "second_order": {"k": k_2nd, "r2": r2_2nd, "k_linear": k_linear_2nd, "r2_linear": r2_linear_2nd} if popt_2nd is not None else None,
            })

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
