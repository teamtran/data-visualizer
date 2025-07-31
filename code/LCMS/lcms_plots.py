import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import scipy.stats as stats
import pandas as pd
import json
import os
import plotly.express as px
from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from sklearn.metrics import r2_score
from matplotlib.ticker import ScalarFormatter
import pdb
import copy


class LCMSPlots:
    """
    Class that contains methods to plot lcms data (multiple lcms curves altogether).
    """

    def __init__(
        self,
        data_dir: Path,
        lcms_data_path: list[Path],
        labels: list[str],
        colors: list[str],
        result_dir: Path,
        style_path: Path,
    ):
        """
        Initialize the class with the data path and the result directory.
        """
        self.data_dir = data_dir
        self.lcms_data_path = lcms_data_path
        self.labels = labels
        self.colors = colors
        self.result_dir = result_dir
        self.style_path = style_path
        self.style = json.load(open(style_path))
        self.result_name = ""
        for label in labels:
            self.result_name += label + "_"

    def preprocess(
        self, lcms_data: pd.DataFrame, normalize_space: np.array, i: int
    ) -> pd.DataFrame:
        """Function that applies transformation to the dataframe which will make it ready for plotting. Note, this is specific to lcms."""
        # Normalize data
        lcms_data[lcms_data.columns[1]] = (
            lcms_data[lcms_data.columns[1]] / lcms_data[lcms_data.columns[1]].max()
        ) * normalize_space[i]
        return lcms_data

    def intensity_at_rt(self, lcms_data: pd.DataFrame, rt: float) -> float:
        """
        Function that returns the intensity at a given retention time.
        """
        # Find the intensity closest to the given retention time
        min_diff_idx = abs(lcms_data[lcms_data.columns[0]] - rt).idxmin()
        intensity = lcms_data.at[min_diff_idx, lcms_data.columns[1]]
        return intensity

    def plot_lcms(
        self,
        lcms_metadata: list[str],
        xlim: tuple,
        ylim: tuple = (-0.1, 1),
        inset_xlim: tuple = None,
        rt=7.25,
        nm: float = 230,
    ):
        """
        Function that plots lcms data.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.tight_layout(pad=3)
        # aesthetics
        ax.set_xlabel("Retention Time (min)", fontsize=12)
        ax.set_ylabel(f"Normalized Intensity at lambda={nm}nm", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
        # Incrementally normalize the data to a lower point
        normalize_upper_limit = 1
        normalize_lower_limit = 1
        normalize_space = np.linspace(
            normalize_upper_limit, normalize_lower_limit, len(self.lcms_data_path)
        )
        i = 0
        # Define the "zoomed in" area
        if inset_xlim is not None:
            axins = ax.inset_axes(
                [0.08, 0.7, 0.4, 0.3]
            )  # [x, y, width, height] of inset axis
            axins.tick_params(axis="both", which="major", labelsize=8, direction="in")
        intensity_at_rt = {}
        for lcms_file, label, color in zip(
            self.lcms_data_path, self.labels, self.colors
        ):
            lcms_data = pd.read_csv(self.data_dir / lcms_file, skiprows=1, sep="\t")
            lcms_data = self.preprocess(lcms_data, normalize_space, i)  # normalize data
            intensity_at_rt[label] = self.intensity_at_rt(lcms_data, rt)
            ax.plot(
                lcms_data[lcms_data.columns[0]],
                lcms_data[lcms_data.columns[1]],
                label=label,
                color=color,
                linewidth=0.2,
            )
            i += 1
            if inset_xlim is not None:
                start_idx = lcms_data[
                    lcms_data[lcms_data.columns[0]] == inset_xlim[0]
                ].index[0]
                end_idx = lcms_data[
                    lcms_data[lcms_data.columns[0]] == inset_xlim[1]
                ].index[0]
                axins.plot(
                    lcms_data[lcms_data.columns[0]][start_idx:end_idx],
                    lcms_data[lcms_data.columns[1]][start_idx:end_idx],
                    color=color,
                    linewidth=0.2,
                )  # Plot the zoomed in area restricted to the start_idx

        if inset_xlim is not None:
            # Set the limits of the zoomed in area
            box, c1 = ax.indicate_inset_zoom(axins, lw=0.4, edgecolor="black", alpha=1)
            c1[0].set_visible(True)
            c1[1].set_visible(True)
            c1[2].set_visible(True)
            c1[3].set_visible(True)
            box.set(linewidth=0.4, alpha=0.8)
            plt.setp([c1[:]], linestyle=":", lw=0.5)
        legend = ax.legend(
            loc="upper right",
            frameon=False,
            title="Experiment Conditions",
            fontsize=8,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # add another legend for lcms data (Mn, Mw, PDI)
        legend_handles = ax.get_legend_handles_labels()[0]
        if lcms_metadata != []:
            lcms_legend = (legend_handles, lcms_metadata)
            lcms_metadata_legend = ax.legend(
                lcms_legend,
                labels=lcms_metadata,
                loc="center right",
                frameon=False,
                title="lcms Data",
                fontsize=8,
            )
        ax.add_artist(legend)
        plt.savefig(self.result_dir / f"{self.result_name}lcms_plot.png", dpi=600)

        # Save intensiy data
        intensity_at_rt_df = pd.DataFrame.from_dict(intensity_at_rt, orient="index")
        intensity_at_rt_df.to_csv(
            self.result_dir / f"{self.result_name}intensity_at_rt.csv"
        )

    def plot_stacked_lcms(
        self,
        lcms_metadata: list[str] = None,
        xlim: tuple = None,
        vertical_spacing: float = 0.5,
        nm: float = 380,
        show_legend: bool = True,
        normalize_individual: bool = True,
    ):
        """
        Function that plots LCMS data as stacked chromatograms with shared x-axis.

        Parameters:
        -----------
        lcms_metadata : list[str], optional
            Metadata labels for each chromatogram
        xlim : tuple, optional
            X-axis limits (min_time, max_time)
        vertical_spacing : float, default=0.3
            Vertical spacing between stacked chromatograms
        nm : float, default=230
            Wavelength for the y-axis label
        show_legend : bool, default=True
            Whether to show the legend
        normalize_individual : bool, default=True
            Whether to normalize each chromatogram individually to [0,1]
        """
        fig, ax = plt.subplots(figsize=(10, 2 + len(self.lcms_data_path) * 1.5))
        plt.tight_layout(pad=3)

        # Aesthetics
        ax.set_xlabel("Retention Time (min)", fontsize=12)
        ax.set_ylabel(f"Chromatograms at λ={nm}nm", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")

        y_positions = []  # Store y-positions for each chromatogram
        max_intensity_global = 0  # For global normalization if needed

        # First pass: determine global max if not normalizing individually
        if not normalize_individual:
            for lcms_file in self.lcms_data_path:
                lcms_data = pd.read_csv(self.data_dir / lcms_file, skiprows=1, sep=",")
                current_max = lcms_data[lcms_data.columns[1]].max()
                max_intensity_global = max(max_intensity_global, current_max)

        # Plot each chromatogram
        for i, (lcms_file, label, color) in enumerate(
            zip(self.lcms_data_path, self.labels, self.colors)
        ):
            lcms_data = pd.read_csv(self.data_dir / lcms_file, skiprows=1, sep=",")

            # Normalize data
            if normalize_individual:
                # Normalize each chromatogram to [0,1]
                normalized_intensity = (
                    lcms_data[lcms_data.columns[1]]
                    / lcms_data[lcms_data.columns[1]].max()
                )
            else:
                # Use global normalization
                normalized_intensity = (
                    lcms_data[lcms_data.columns[1]] / max_intensity_global
                )

            # Calculate vertical offset for stacking
            y_offset = i * vertical_spacing
            y_positions.append(y_offset)

            # Plot the chromatogram
            ax.plot(
                lcms_data[lcms_data.columns[0]],
                normalized_intensity + y_offset,
                label=label,
                color=color,
                linewidth=1.0,
            )

            # Add a horizontal baseline for each chromatogram
            if xlim:
                baseline_x = xlim
            else:
                baseline_x = (
                    lcms_data[lcms_data.columns[0]].min(),
                    lcms_data[lcms_data.columns[0]].max(),
                )

            ax.plot(
                baseline_x,
                [y_offset, y_offset],
                color="gray",
                linewidth=0.5,
                alpha=0.5,
                linestyle="--",
            )

            # Add label on the left side of each chromatogram
            ax.text(
                -0.02,
                y_offset + 0.5,
                label,
                transform=ax.get_yaxis_transform(),
                horizontalalignment="right",
                verticalalignment="center",
                fontsize=10,
                color=color,
                weight="bold",
            )

        # Set axis limits
        if xlim:
            ax.set_xlim(xlim)

        # Set y-axis limits with some padding
        y_min = -0.1
        y_max = max(y_positions) + 1.2
        ax.set_ylim(y_min, y_max)

        # Customize y-axis - remove ticks since they're not meaningful for stacked plot
        ax.set_yticks([])

        # Add legend if requested
        if show_legend:
            legend = ax.legend(
                loc="upper right",
                frameon=False,
                title="Chromatograms",
                fontsize=10,
            )

        # Add metadata legend if provided
        if lcms_metadata and lcms_metadata != []:
            legend_handles = ax.get_legend_handles_labels()[0]
            lcms_legend = (legend_handles, lcms_metadata)
            lcms_metadata_legend = ax.legend(
                lcms_legend[0],
                labels=lcms_metadata,
                loc="center right",
                frameon=False,
                title="LCMS Data",
                fontsize=8,
            )
            if show_legend:
                ax.add_artist(legend)

        # Save the plot
        plt.savefig(
            self.result_dir / f"{self.result_name}stacked_lcms_plot.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.savefig(
            self.result_dir / f"{self.result_name}stacked_lcms_plot.svg",
            dpi=600,
            bbox_inches="tight",
        )


class LCMS_LinearCalibration_Plots:
    def __init__(
        self,
        data_dir: Path,
        lcms_filename: str,
        label: str,
        color: str,
        result_dir: Path,
        style_path: Path,
        nm: float,
    ):
        self.data_dir = data_dir
        self.lcms_filename = lcms_filename
        self.label = label
        self.color = color
        self.result_dir = result_dir
        self.style_path = style_path
        self.style = json.load(open(style_path))
        self.result_name = ""
        self.result_name += label + "_"
        self.nm = nm

    def plot_calibration_curve(self):
        data = pd.read_csv(self.data_dir / self.lcms_filename)
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.tight_layout(pad=3)
        ax.set_xlabel("Concentration (mg/mL)", fontsize=12)
        ax.set_ylabel(f"Peak Area at {self.nm}nm", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
        ax.scatter(
            data["Concentration (mg/mL)"],
            data[f"Peak area"],
            label=self.label,
            color=self.color,
        )
        # plot line of best fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            data["Concentration (mg/mL)"], data[f"Peak area"]
        )
        line = slope * data["Concentration (mg/mL)"] + intercept
        r2 = r2_score(data[f"Peak area"], line)
        ax.plot(
            data["Concentration (mg/mL)"],
            line,
            label=f"y = {slope:.4f}x + {intercept:.4f}, R$^2$={r2:.4f}",
            color=self.color,
            linestyle="--",
        )
        ax.legend(loc="upper left", frameon=False)
        ax.set_title(f"Calibration Curve of {self.label} at {self.nm}nm")
        plt.savefig(
            self.result_dir / f"{self.result_name}calibration_curve.png", dpi=600
        )
