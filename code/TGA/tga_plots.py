from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import scipy.stats as stats
from scipy.stats import linregress
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
        new_columns[3] = "Mass loss/pct"
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
            print(f"Warning: Could not process MS data - {e}")
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
        fig, ax = plt.subplots(1, figsize=(4, 3.25))
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
                self.data_dir / tga_path, encoding="iso-8859-1", on_bad_lines="warn"
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
        t_depolymerization_cutoff: float = 99,
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
            print(tga_path)
            tga_data = pd.read_csv(
                self.data_dir / tga_path,
                encoding="iso-8859-1",
                on_bad_lines="warn",
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
            print(f"{label}: {t_depolymerization_temp}")
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
                label=label + f" (T{'$_{onset}$'} = {onset_temp:.1f}°C)",
                color=color,
                zorder=5,
            )

        # Draw a line in the y-axis at y=99%
        ax.axhline(
            y=t_depolymerization_cutoff,
            color="r",
            linestyle="--",
            linewidth=0.3,
            label="$T_\mathregular{depolymerization}$"
            + f" at {t_depolymerization_cutoff}% Mass",
        )

        ax.legend(fontsize=10)
        ax.set_xlim(xlim)
        ax.set_yticks([0, 20, 40, 60, 80, 100])  # Add this line
        ax.set_ylim(0, 100)
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
        ax.set_ylabel(
            "Normalized Peak Area of Ion Current over Time (A x min)", fontsize=5
        )
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
            color="#A0A0A0",
            marker="o",
            alpha=0.7,
            label="Pristine PS",
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
            color="#A0A0A0",
            marker="o",
            alpha=0.7,
            label="Pristine PS",
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
        self, summary_dir, ylim=(310, 360), xlim=(-0.5, 6)
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

        print(df_functionalized)

        # Clean data for onset temperature (remove NaN values)
        df_pristine_onset = df_pristine.dropna(
            subset=["percent_functionalization", "dynamic_onset_t"]
        )
        df_func_onset = df_functionalized.dropna(
            subset=["percent_functionalization", "dynamic_onset_t"]
        )

        print(df_func_onset)

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
            color="#A0A0A0",
            marker="o",
            alpha=0.7,
            label="Pristine PS",
            linewidth=1,
        )

        ax.scatter(
            df_func_onset["percent_functionalization"],
            df_func_onset["dynamic_onset_t"],
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
            self.result_dir / f"onset_temperature_vs_percent_functionalization.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir / f"onset_temperature_vs_percent_functionalization.eps",
            format="eps",
            dpi=400,
        )

    def plot_mass_loss_vs_percent_functionalization(
        self, summary_dir, ylim=(40, 80), xlim=(-0.5, 6)
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
            color="#A0A0A0",
            marker="o",
            alpha=0.7,
            label="Pristine PS",
            linewidth=1,
        )

        ax.scatter(
            df_func_mass["percent_functionalization"],
            df_func_mass["isothermal_mass_loss_after_1200mins"],
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
        virgin_colors = ["#808080", "#A0A0A0", "#606060", "#909090", "#707070"]

        mass_difference_at_time = []

        # Plot virgin PS samples in gray
        # for i, (tga_path, label) in enumerate(virgin_ps_data_paths):
        #     color = virgin_colors[i % len(virgin_colors)]

        #     # Read and process TGA data
        #     tga_data = pd.read_csv(
        #         self.data_dir / tga_path, encoding="iso-8859-1", on_bad_lines="warn"
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
                self.data_dir / tga_path, encoding="iso-8859-1", on_bad_lines="warn"
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
        # virgin_colors = ["#808080", "#A0A0A0", "#606060", "#909090", "#707070"]

        # # Plot virgin PS samples in gray
        # for i, (tga_path, label) in enumerate(virgin_ps_data_paths):
        #     color = virgin_colors[i % len(virgin_colors)]

        #     # Read and process TGA data
        #     tga_data = pd.read_csv(
        #         self.data_dir / tga_path,
        #         encoding="iso-8859-1",
        #         on_bad_lines="warn",
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
                on_bad_lines="warn",
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
