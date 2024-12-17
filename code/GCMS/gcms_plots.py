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


class GCMSPlots:
    """
    Class that contains methods to plot gcms chromatogram (multiple gcms curves altogether) and individual MS spectra.
    """

    def __init__(
        self,
        data_dir: Path,
        gcms_data_path: list[Path],
        gcms_type: str,
        labels: list[str],
        colors: list[str],
        result_dir: Path,
        style_path: Path,
    ):
        """
        Initialize the class with the data path and the result directory.
        """
        self.data_dir = data_dir
        self.gcms_data_path = gcms_data_path
        self.gcms_type = gcms_type  # XIC_at_320mz, TIC
        self.labels = labels
        self.colors = colors
        self.result_dir = result_dir
        self.style_path = style_path
        self.style = json.load(open(style_path))
        self.result_name = ""
        for label in labels:
            self.result_name += label + "_"

    def preprocess(self, gcms_data: pd.DataFrame) -> pd.DataFrame:
        """Function that applies transformation to the dataframe which will make it ready for plotting. Note, this is specific to gcms."""
        # Normalize data
        gcms_data[gcms_data.columns[2]] = (
            gcms_data[gcms_data.columns[2]] / gcms_data[gcms_data.columns[2]].max()
        )
        return gcms_data

    def intensity_at_rt(self, gcms_data: pd.DataFrame, rt: float) -> float:
        """
        Function that returns the intensity at a given retention time.
        """
        # Find the intensity closest to the given retention time
        min_diff_idx = abs(gcms_data[gcms_data.columns[0]] - rt).idxmin()
        intensity = gcms_data.at[min_diff_idx, gcms_data.columns[1]]
        return intensity

    def plot_gcms(
        self,
        gcms_metadata: list[str],
        xlim: tuple,
        ylim: tuple = (-0.1, 1),
    ):
        """
        Function that plots gcms data.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.tight_layout(pad=3)
        # aesthetics
        ax.set_xlabel("Retention Time (min)", fontsize=12)
        ax.set_ylabel(f"Normalized Intensity at lambda={nm}nm", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
        for gcms_file, label, color in zip(
            self.gcms_data_path, self.labels, self.colors
        ):
            gcms_data = pd.read_csv(self.data_dir / gcms_file, skiprows=42, sep="\t")
            gcms_data = self.preprocess(gcms_data)  # normalize data
            ax.plot(
                gcms_data[gcms_data.columns[0]],
                gcms_data[gcms_data.columns[1]],
                label=label,
                color=color,
                linewidth=0.2,
            )
            i += 1
        legend = ax.legend(
            loc="upper right",
            frameon=False,
            title=f"Sample ID ({self.gcms_type})",
            fontsize=8,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.add_artist(legend)
        plt.savefig(
            self.result_dir / f"{self.result_name}{self.gcms_type}_gcms_plot.png",
            dpi=600,
        )


class MSPlots:
    def __init__(
        self,
        data_dir: Path,
        ms_data_path: list[Path],
        result_dir: Path,
        style_path: Path,
    ):
        """
        Initialize the class with the data path and the result directory.
        """
        self.data_dir = data_dir
        self.ms_data_path = ms_data_path
        self.result_dir = result_dir
        self.style_path = style_path
        self.style = json.load(open(style_path))
        self.result_name = ""

    def preprocess(self, ms_data: pd.DataFrame) -> pd.DataFrame:
        """Function that applies transformation to the dataframe which will make it ready for plotting. Note, this is specific to gcms."""
        # Normalize data
        ms_data[ms_data.columns[1]] = (
            ms_data[ms_data.columns[1]] / ms_data[ms_data.columns[1]].max()
        )
        return ms_data

    def plot_ms(
        self,
        time: float,
        xlim: tuple,
        ylim: tuple = (-0.1, 1),
    ):
        """
        Function that plots gcms data.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.tight_layout(pad=3)
        # aesthetics
        ax.set_xlabel("Charge to mass ratio (m/z)", fontsize=12)
        ax.set_ylabel(f"Normalized Ion Intensity at {time}min", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")

        gcms_data = pd.read_csv(
            self.data_dir / self.ms_data_path, skiprows=35, sep="\t"
        )
        gcms_data = self.preprocess(gcms_data)  # normalize data
        ax.plot(
            gcms_data[gcms_data.columns[0]],
            gcms_data[gcms_data.columns[1]],
            linewidth=0.2,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        time = str(time).replace(".", "_")
        plt.savefig(
            self.result_dir / f"{self.result_name}ms_spectrum_at_{time}min.png",
            dpi=600,
        )


class GCMS_LinearCalibration_Plots:
    def __init__(
        self,
        data_dir: Path,
        gcms_filename: str,
        label: str,
        color: str,
        result_dir: Path,
        style_path: Path,
        rt: float,
    ):
        self.data_dir = data_dir
        self.gcms_filename = gcms_filename
        self.label = label
        self.color = color
        self.result_dir = result_dir
        self.style_path = style_path
        self.style = json.load(open(style_path))
        self.result_name = ""
        self.result_name += label + "_"
        self.rt = rt

    def plot_calibration_curve(self):
        data = pd.read_csv(self.data_dir / self.gcms_filename)
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
        ax.set_title(f"Calibration Curve of {self.label} at rt={self.rt}min")
        plt.savefig(
            self.result_dir / f"{self.result_name}calibration_curve.png", dpi=600
        )
