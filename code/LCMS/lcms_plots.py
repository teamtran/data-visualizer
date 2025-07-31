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
