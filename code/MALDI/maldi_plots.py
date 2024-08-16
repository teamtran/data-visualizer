from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import scipy.signal
import scipy.stats as stats
import pandas as pd
import json
import os
import scipy
from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from sklearn.metrics import r2_score
from matplotlib.ticker import ScalarFormatter
import pdb

# TODO: maldi plot, peak detection, labelling nicely, stacked plots, normalization, 3D plot (hplc, maldi)


class MALDIPlots:
    """
    Class that contains methods to plot TGA-MS (dynamic, isothermal) data.
    """

    def __init__(
        self,
        data_dir: Path,
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
        ms_data: pd.DataFrame,
        xlim: tuple = (600, 6000),
        normalize: bool = True,
    ):
        """
        Preprocess MS data by performing peak detection, labelling peaks, and normalizing the data.
        """
        # filter data by xlim
        ms_data = ms_data[(ms_data["m/z"] >= xlim[0]) & (ms_data["m/z"] <= xlim[1])]
        print(ms_data)
        if normalize:
            ms_data["Intensity"] = ms_data["Intensity"] / ms_data["Intensity"].max()

        return ms_data

    def plot_maldi(
        self,
        xlim: tuple = (600, 6000),
        ylim: tuple = (-0.1, 1),
        normalize: bool = True,
    ):
        """
        Plot several TGA isothermal data for comparison (can handle 1 or more).
        """
        fig, ax = plt.subplots(1, figsize=(6, 6))
        plt.subplots_adjust(hspace=0.5)
        # aesthetics
        ax.set_xlabel("m/z", fontsize=12)
        if normalize:
            ax.set_ylabel("Normalized Intensity", fontsize=12)
        else:
            ax.set_ylabel("Intensity", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
        # set xlim for the plots
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        title: str = ""
        for ms_path, label, color in zip(self.ms_data_path, self.labels, self.colors):
            title += label + "_"
            ms_data = pd.read_csv(self.data_dir / ms_path, sep=" ", header=None)
            ms_data = ms_data.rename(columns={0: "m/z", 1: "Intensity"})
            ms_data = self.preprocess(ms_data, xlim=xlim, normalize=True)
            ax.plot(ms_data["m/z"], ms_data["Intensity"], label=label, color=color)
        ax.set_title(title + "MALDI", pad=20, fontsize=16)
        ax.legend()
        plt.savefig(
            self.result_dir / f"{self.result_name}maldi.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir / f"{self.result_name}maldi.svg",
            dpi=400,
        )

    def plot_maldi_zoom(
        self,
        xlim: tuple = (600, 6000),
        ylim: tuple = (-0.1, 1),
        peak_detection: bool = True,
        normalize: bool = True,
        prominence: float = 0.05,
    ):
        fig, ax = plt.subplots(1, figsize=(6, 6))
        plt.subplots_adjust(hspace=0.5)
        # aesthetics
        ax.set_xlabel("m/z", fontsize=12)
        if normalize:
            ax.set_ylabel("Normalized Intensity", fontsize=12)
        else:
            ax.set_ylabel("Intensity", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
        # set xlim for the plots
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        title: str = ""
        for ms_path, label, color in zip(self.ms_data_path, self.labels, self.colors):
            title += label + "_"
            ms_data = pd.read_csv(self.data_dir / ms_path, sep=" ", header=None)
            ms_data = ms_data.rename(columns={0: "m/z", 1: "Intensity"})
            ms_data = self.preprocess(ms_data, xlim=xlim, normalize=True)
            # peak detection
            if peak_detection:
                peak_idx, properties = scipy.signal.find_peaks(
                    ms_data["Intensity"], prominence=prominence
                )
            ax.plot(ms_data["m/z"], ms_data["Intensity"], label=label, color=color)
            ax.plot(
                ms_data.iloc[peak_idx]["m/z"], ms_data.iloc[peak_idx]["Intensity"], "xr"
            )
            for i, peak in enumerate(peak_idx):
                ax.text(
                    ms_data.iloc[peak]["m/z"],
                    ms_data.iloc[peak]["Intensity"] + 0.02,
                    f"{ms_data.iloc[peak]['m/z']:.2f}",
                    fontsize=8,
                )
        ax.set_title(title + "MALDI", pad=20, fontsize=16)
        ax.legend()
        plt.savefig(
            self.result_dir / f"{self.result_name}maldi_zoom_{xlim[0]}-{xlim[1]}.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir / f"{self.result_name}maldi_zoom_{xlim[0]}-{xlim[1]}.svg",
            dpi=400,
        )
