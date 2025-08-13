import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import scipy
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
        # Baseline correction according to initial 50 datapoints
        gcms_data[gcms_data.columns[2]] = (
            gcms_data[gcms_data.columns[2]]
            - gcms_data[gcms_data.columns[2]].iloc[:50].mean()
        )
        # Normalize data
        print(f"{gcms_data[gcms_data.columns[2]]=}")
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
        xlim: tuple,
        ylim: tuple = (-0.05, 1),
    ):
        """
        Function that plots gcms data.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.tight_layout(pad=3)
        # aesthetics
        ax.set_xlabel("Retention Time (min)", fontsize=12)
        ax.set_ylabel(f"Normalized Intensity of Ion Count", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
        for gcms_file, label, color in zip(
            self.gcms_data_path, self.labels, self.colors
        ):
            # remove commas from text file
            with open(self.data_dir / gcms_file, "r") as f:
                lines = f.readlines()
            with open(self.data_dir / gcms_file, "w") as f:
                for line in lines:
                    f.write(line.replace(",", ""))
            gcms_data = pd.read_csv(self.data_dir / gcms_file, skiprows=42, sep="\t")
            gcms_data.dropna(inplace=True)
            gcms_data = self.preprocess(gcms_data)  # normalize data
            ax.plot(
                gcms_data[gcms_data.columns[0]],
                gcms_data[gcms_data.columns[2]],
                label=label,
                color=color,
                linewidth=1,
            )
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
        plt.savefig(
            self.result_dir / f"{self.result_name}{self.gcms_type}_gcms_plot.eps",
            format="eps",
            dpi=600,
        )


class MSPlots:
    def __init__(
        self,
        data_dir: Path,
        ms_data_path: str,
        result_dir: Path,
        style_path: Path,
        label: str,
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
        self.result_name += label + "_"

    def preprocess(self, ms_data: pd.DataFrame, xlim: tuple) -> pd.DataFrame:
        """Function that applies transformation to the dataframe which will make it ready for plotting. Note, this is specific to gcms."""
        # filter data by xlim
        ms_data = ms_data[
            (ms_data[ms_data.columns[0]] >= xlim[0])
            & (ms_data[ms_data.columns[0]] <= xlim[1])
        ]

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
        prominence: float = 0.01,
        inset_xlim: tuple = (340, 380),
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
        # remove commas from text file
        with open(self.data_dir / self.ms_data_path, "r") as f:
            lines = f.readlines()
        with open(self.data_dir / self.ms_data_path, "w") as f:
            for line in lines:
                f.write(line.replace(",", ""))
        gcms_data = pd.read_csv(
            self.data_dir / self.ms_data_path, skiprows=35, sep="\t"
        )
        gcms_data = self.preprocess(gcms_data, xlim)  # normalize data
        ax.stem(
            gcms_data[gcms_data.columns[0]],
            gcms_data[gcms_data.columns[1]],
            markerfmt=" ",
            basefmt=" ",
        )
        # label peaks higher than 0.02
        peak_idx, properties = scipy.signal.find_peaks(
            gcms_data[gcms_data.columns[1]], prominence=prominence
        )
        for i, peak in enumerate(peak_idx):
            # Write text rotated 45 degrees.

            ax.text(
                gcms_data.iloc[peak][gcms_data.columns[0]],
                gcms_data.iloc[peak][gcms_data.columns[1]] + 0.02,
                f"{gcms_data.iloc[peak][gcms_data.columns[0]]:.2f}",
                fontsize=8,
                rotation=30,
            )

        # Define the "zoomed in" area
        if inset_xlim is not None:
            axins = ax.inset_axes(
                [0.6, 0.4, 0.4, 0.3]
            )  # [x, y, width, height] of inset axis
            axins.spines["top"].set_visible(False)
            axins.spines["right"].set_visible(False)
            axins.tick_params(axis="both", which="major", labelsize=8, direction="in")
            start_idx = gcms_data[
                gcms_data[gcms_data.columns[0]] >= inset_xlim[0]
            ].index[0]
            end_idx = gcms_data[gcms_data[gcms_data.columns[0]] >= inset_xlim[1]].index[
                0
            ]
            axins.stem(
                gcms_data[gcms_data.columns[0]][start_idx:end_idx],
                gcms_data[gcms_data.columns[1]][start_idx:end_idx],
                markerfmt=" ",
                basefmt=" ",
            )  # Plot the zoomed in area restricted to the start_idx
            # label peaks higher than 0.02
            peak_idx, properties = scipy.signal.find_peaks(
                gcms_data[gcms_data.columns[1]][start_idx:end_idx],
                prominence=0.000001,
            )
            print(f"{peak_idx=}")
            for i, peak in enumerate(peak_idx):
                # Write text rotated 45 degrees.
                axins.text(
                    gcms_data[gcms_data.columns[0]][start_idx:end_idx].iloc[peak],
                    gcms_data[gcms_data.columns[1]][start_idx:end_idx].iloc[peak]
                    + 0.0005,
                    f"{gcms_data[gcms_data.columns[0]][start_idx:end_idx].iloc[peak]:.2f}",
                    fontsize=8,
                    rotation=30,
                )
            # Set the limits of the zoomed in area
            box, c1 = ax.indicate_inset_zoom(axins, lw=0.4, edgecolor="black", alpha=1)
            c1[0].set_visible(True)
            c1[1].set_visible(True)
            c1[2].set_visible(True)
            c1[3].set_visible(True)
            box.set(linewidth=0.4, alpha=0.8)
            plt.setp([c1[:]], linestyle=":", lw=0.5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        time = str(time).replace(".", "_")
        plt.savefig(
            self.result_dir / f"{self.result_name}ms_spectrum_at_{time}min.png",
            dpi=600,
        )
        plt.savefig(
            self.result_dir / f"{self.result_name}ms_spectrum_at_{time}min.eps",
            format="eps",
            dpi=600,
        )

    def plot_ms_zoom(
        self,
        time: float,
        xlim: tuple,
        ylim: tuple = (-0.1, 1),
        prominence: float = 0.002,
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
        # remove commas from text file
        with open(self.data_dir / self.ms_data_path, "r") as f:
            lines = f.readlines()
        with open(self.data_dir / self.ms_data_path, "w") as f:
            for line in lines:
                f.write(line.replace(",", ""))
        gcms_data = pd.read_csv(
            self.data_dir / self.ms_data_path, skiprows=35, sep="\t"
        )
        gcms_data = self.preprocess(gcms_data, xlim)  # normalize data
        ax.stem(
            gcms_data[gcms_data.columns[0]],
            gcms_data[gcms_data.columns[1]],
            markerfmt=" ",
            basefmt=" ",
        )
        # label peaks higher than 0.02
        peak_idx, properties = scipy.signal.find_peaks(
            gcms_data[gcms_data.columns[1]], prominence=prominence
        )
        for i, peak in enumerate(peak_idx):
            ax.text(
                gcms_data.iloc[peak][gcms_data.columns[0]],
                gcms_data.iloc[peak][gcms_data.columns[1]] + 0.02,
                f"{gcms_data.iloc[peak][gcms_data.columns[0]]:.2f}",
                fontsize=8,
                rotation=30,
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        time = str(time).replace(".", "_")
        plt.savefig(
            self.result_dir
            / f"{self.result_name}ms_spectrum_at_{time}min_{xlim[0]}_{xlim[1]}m_z_zoom.png",
            dpi=600,
        )
        plt.savefig(
            self.result_dir
            / f"{self.result_name}ms_spectrum_at_{time}min_{xlim[0]}_{xlim[1]}m_z_zoom.eps",
            format="eps",
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
