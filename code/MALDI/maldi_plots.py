from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import scipy.signal
import scipy.stats as stats
import pandas as pd
import plotly.express as px
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
        opacity: list[float],
    ):
        """
        Initialize the class with the data path and the result directory.
        """
        self.data_dir = data_dir
        self.ms_data_path = ms_data_path
        self.labels = labels
        self.colors = colors
        self.opacity = opacity
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
        # print(ms_data)
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
        fig, ax = plt.subplots(1, figsize=(6, 4))
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
        for ms_path, label, color, opacity in zip(
            self.ms_data_path, self.labels, self.colors, self.opacity
        ):
            title += label + "_"
            ms_data = pd.read_csv(self.data_dir / ms_path, sep=" ", header=None)
            ms_data = ms_data.rename(columns={0: "m/z", 1: "Intensity"})
            ms_data = self.preprocess(ms_data, xlim=xlim, normalize=True)
            ax.plot(
                ms_data["m/z"],
                ms_data["Intensity"],
                label=label,
                color=color,
                linewidth=0.5,
                alpha=opacity,
            )
        ax.set_title(title + "MALDI", pad=20, fontsize=16)
        ax.legend()
        plt.savefig(
            self.result_dir / f"{self.result_name}maldi.png",
            dpi=300,
        )
        plt.savefig(
            self.result_dir / f"{self.result_name}maldi.svg",
            dpi=300,
        )

    def plot_maldi_zoom(
        self,
        xlim: tuple = (600, 6000),
        ylim: tuple = (-0.1, 1),
        peak_detection: bool = True,
        normalize: bool = True,
        prominence: float = 0.05,
    ):
        fig, ax = plt.subplots(1, figsize=(6, 2.5))
        plt.tight_layout(pad=3.5, w_pad=0.5, h_pad=0.5)
        # plt.subplots_adjust(wspace=0.5)
        # aesthetics
        ax.set_xlabel("m/z", fontsize=12)
        if normalize:
            ax.set_ylabel("Normalized \n Intensity", fontsize=12)
        else:
            ax.set_ylabel("Intensity", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
        # set xlim for the plots
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        title: str = ""
        for ms_path, label, color, opacity in zip(
            self.ms_data_path, self.labels, self.colors, self.opacity
        ):
            title += label + "_"
            ms_data = pd.read_csv(self.data_dir / ms_path, sep=" ", header=None)
            ms_data = ms_data.rename(columns={0: "m/z", 1: "Intensity"})
            ms_data = self.preprocess(ms_data, xlim=xlim, normalize=True)
            # peak detection
            if peak_detection:
                peak_idx, properties = scipy.signal.find_peaks(
                    ms_data["Intensity"], prominence=prominence
                )
            ax.plot(
                ms_data["m/z"],
                ms_data["Intensity"],
                label=label,
                color=color,
                linewidth=0.6,
                alpha=opacity,
            )
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
        # ax.set_title(title + "MALDI", pad=20, fontsize=16)
        ax.legend(loc="upper center")
        plt.savefig(
            self.result_dir / f"{self.result_name}maldi_zoom_{xlim[0]}-{xlim[1]}.png",
            dpi=300,
        )
        plt.savefig(
            self.result_dir / f"{self.result_name}maldi_zoom_{xlim[0]}-{xlim[1]}.eps",
            format="eps",
            dpi=300,
        )

    def plot_maldi_zoom_with_inset(
        self,
        xlim: tuple = (600, 6000),
        ylim: tuple = (-0.1, 1),
        peak_detection: bool = True,
        normalize: bool = True,
        prominence: float = 0.05,
        inset_xlim: tuple = None,
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
        # Define the "zoomed in" area
        if inset_xlim is not None:
            axins = ax.inset_axes(
                [0.35, 0.6, 0.4, 0.3]
            )  # [x, y, width, height] of inset axis
            axins.tick_params(axis="both", which="major", labelsize=8, direction="in")
        for ms_path, label, color, opacity in zip(
            self.ms_data_path, self.labels, self.colors, self.opacity
        ):
            title += label + "_"
            ms_data = pd.read_csv(self.data_dir / ms_path, sep=" ", header=None)
            ms_data = ms_data.rename(columns={0: "m/z", 1: "Intensity"})
            ms_data = self.preprocess(ms_data, xlim=xlim, normalize=True)

            # peak detection
            if peak_detection:
                peak_idx, properties = scipy.signal.find_peaks(
                    ms_data["Intensity"], prominence=prominence
                )
            ax.plot(
                ms_data["m/z"],
                ms_data["Intensity"],
                label=label,
                color=color,
                linewidth=0.6,
                alpha=opacity,
            )
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
            if inset_xlim is not None:
                start_idx = ms_data[ms_data[ms_data.columns[0]] >= inset_xlim[0]].index[
                    0
                ]
                end_idx = ms_data[ms_data[ms_data.columns[0]] >= inset_xlim[1]].index[0]
                # print(f"{ms_data.loc[start_idx:end_idx]=}")
                axins.plot(
                    ms_data[ms_data.columns[0]].loc[start_idx:end_idx],
                    ms_data[ms_data.columns[1]].loc[start_idx:end_idx],
                    color=color,
                    linewidth=0.4,
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

    def plotly_maldi_zoom(
        self,
        xlim: tuple = (600, 6000),
        ylim: tuple = (-0.1, 1),
        peak_detection: bool = True,
        normalize: bool = True,
    ):
        title: str = ""
        ms_data_tidy_list = []
        for ms_path, label, color in zip(self.ms_data_path, self.labels, self.colors):
            # concatenate the data into tidy data format
            title += label + "_"
            ms_data = pd.read_csv(self.data_dir / ms_path, sep=" ", header=None)
            ms_data = ms_data.rename(columns={0: "m/z", 1: "Intensity"})
            ms_data = self.preprocess(ms_data, xlim=xlim, normalize=True)
            # Reduce amount of data to prevent lagging on plotly
            print(len(ms_data))
            ms_data = ms_data.iloc[::5, :]
            print(len(ms_data))
            ms_data["label"] = label
            ms_data_tidy_list.append(ms_data)
        # concatenate all dataframes
        ms_data_tidy = pd.concat(ms_data_tidy_list)
        fig = px.line(
            ms_data_tidy,
            x="m/z",
            y="Intensity",
            color="label",
            title=title + "MALDI",
        )
        fig.update_layout(yaxis_range=[ylim[0], ylim[1]])
        fig.update_layout(xaxis_range=[xlim[0], xlim[1]])
        return fig
