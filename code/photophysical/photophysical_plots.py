import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import scipy.stats as stats
import pandas as pd
import json
import os
from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from sklearn.metrics import r2_score
from matplotlib.ticker import ScalarFormatter
import pdb
import copy


class PhotophysicalPlots:
    """
    Class that contains methods to plot uv_vis data (multiple uv_vis curves altogether).
    """

    def __init__(
        self,
        data_dir: Path,
        labels: list[str],
        colors: list[str],
        result_dir: Path,
        style_path: Path,
        uv_vis_data_path: Path = None,
        photoluminescence_data_path: Path = None,
        uv_vis_experiment_names: list[str] = None,
        photoluminescence_experiment_names: list[str] = None,
    ):
        """
        Initialize the class with the data path and the result directory.
        """
        self.data_dir = data_dir
        self.uv_vis = False
        self.pl = False
        if uv_vis_data_path is not None:
            self.uv_vis_data_path = data_dir / uv_vis_data_path
            self.uv_vis = True
        if photoluminescence_data_path is not None:
            self.photoluminescence_data_path = data_dir / photoluminescence_data_path
            self.pl = True
        if uv_vis_experiment_names is not None:
            self.uv_vis_experiment_names = uv_vis_experiment_names
        if photoluminescence_experiment_names is not None:
            self.photoluminescence_experiment_names = photoluminescence_experiment_names
        self.labels = labels
        self.colors = colors
        self.result_dir = result_dir
        self.style_path = style_path
        self.style = json.load(open(style_path))
        self.result_name = ""
        for label in labels:
            self.result_name += label + "_"
        # BIG TODO: Make sure to be able to handle plotting 1) uv-vis, 2) PL, and 3) uv-vis + pl

    def preprocess(
        self,
        drop_columns: list[int],
        normalize: bool,
        baseline: bool,
        xlim: tuple,
    ) -> pd.DataFrame:
        """Function that applies transformation to the dataframe which will make it ready for plotting. Note, this is specific to photophysical."""
        if self.uv_vis:
            # Get all index of columns
            temp_data = pd.read_csv(self.uv_vis_data_path)
            columns = range(0, len(temp_data.columns))
            # Drop columns
            use_columns = [x for x in columns if x not in drop_columns]
            # Drop all rows after the data (i.e. drop the metadata)
            # Find row with "Baseline 100%T"
            row_to_drop = temp_data[temp_data.iloc[:, 0] == "Baseline 100%T"].index[0]
            num_of_rows = len(temp_data.index)
            skiprows = list(range(row_to_drop, num_of_rows + 1))
            skiprows.append(1)  # skip row with wavelength and absorbance
            uv_vis_data = pd.read_csv(
                self.uv_vis_data_path, usecols=use_columns, skiprows=skiprows
            )
            # cut off data according to xlim
            # find index of wavelength closest to xlim
            idx_xlim_low = (np.abs(uv_vis_data.iloc[:, 0] - xlim[0])).idxmin()
            idx_xlim_high = (np.abs(uv_vis_data.iloc[:, 0] - xlim[1])).idxmin()
            for expt in self.uv_vis_experiment_names:
                # get all column_idx with expt name
                column_idx = [
                    i + 1
                    for i, column in enumerate(uv_vis_data.columns)
                    if expt in column
                ]
                for col_idx in column_idx:
                    # Baseline correction
                    if baseline:
                        absorbance = uv_vis_data.iloc[
                            idx_xlim_high:idx_xlim_low, col_idx
                        ]
                        min_absorbance = min(absorbance)
                        uv_vis_data.iloc[idx_xlim_high:idx_xlim_low, col_idx] = (
                            absorbance - min_absorbance
                        )
                    # Normalize data # TODO: normalize to lambda max
                    if normalize:  # divide by max absorbance
                        absorbance = uv_vis_data.iloc[
                            idx_xlim_high:idx_xlim_low, col_idx
                        ]
                        max_absorbance = max(absorbance)
                        uv_vis_data.iloc[idx_xlim_high:idx_xlim_low, col_idx] = (
                            absorbance / max_absorbance
                        )
                # average triplicates
                uv_vis_data[expt + "_wavelength"] = uv_vis_data.iloc[
                    idx_xlim_high:idx_xlim_low, col_idx - 1
                ]  # wavelength from last replicate
                uv_vis_data[expt + "_avg"] = uv_vis_data.iloc[
                    idx_xlim_high:idx_xlim_low, column_idx
                ].mean(axis=1)
        if self.pl:
            # photoluminescence data
            pl_data = pd.read_excel(self.photoluminescence_data_path, skiprows=[1])
            # find index of wavelength closest to xlim
            idx_xlim_low = (np.abs(pl_data.iloc[:, 0] - xlim[0])).idxmin()
            idx_xlim_high = (np.abs(pl_data.iloc[:, 0] - xlim[1])).idxmin()
            for expt in self.photoluminescence_experiment_names:
                # get all column_idx with expt name
                column_idx = [
                    i for i, column in enumerate(pl_data.columns) if expt in column
                ]
                for col_idx in column_idx:
                    # Baseline correction
                    if baseline:
                        absorbance = pl_data.iloc[idx_xlim_low:idx_xlim_high, col_idx]
                        min_absorbance = min(absorbance)
                        pl_data.iloc[idx_xlim_low:idx_xlim_high, col_idx] = (
                            absorbance - min_absorbance
                        )
                    # Normalize data
                    if normalize:  # divide by max absorbance
                        absorbance = pl_data.iloc[idx_xlim_low:idx_xlim_high, col_idx]
                        max_absorbance = max(absorbance)
                        pl_data.iloc[idx_xlim_low:idx_xlim_high, col_idx] = (
                            absorbance / max_absorbance
                        )
                # average triplicates
                pl_data[expt + "_wavelength"] = pl_data.iloc[
                    idx_xlim_low:idx_xlim_high, col_idx - 1
                ]  # wavelength from last replicate
                pl_data[expt + "_avg"] = pl_data.iloc[
                    idx_xlim_low:idx_xlim_high, column_idx
                ].mean(axis=1)
        if self.uv_vis and self.pl:
            return uv_vis_data, pl_data
        elif self.uv_vis:
            return uv_vis_data, None
        elif self.pl:
            return None, pl_data

    def plot_uv_vis_and_pl(
        self,
        drop_columns: list[int],
        normalize: bool,
        baseline: bool,
        xlim: tuple,
        ylim: tuple = (-0.1, 1),
    ):
        """
        Function that plots photophysical data.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.tight_layout(pad=5)
        # uv_vis plot
        ax.set_xlabel("Wavelength (nm)", fontsize=self.style["fontsize"])
        ax.set_ylabel("Normalized Absorbance (a.u.)", fontsize=self.style["fontsize"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(
            axis="both", which="major", labelsize=self.style["fontsize"], direction="in"
        )
        # pl plot
        ax2 = ax.twinx()
        ax2.set_ylabel(
            "Normalized PL Intensity (a.u.)", fontsize=self.style["fontsize"]
        )
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.tick_params(
            axis="both", which="major", labelsize=self.style["fontsize"], direction="in"
        )
        i = 0

        uv_vis_data, pl_data = self.preprocess(drop_columns, normalize, baseline, xlim)

        for uv_vis_expt_name, pl_expt_name, label, color in zip(
            self.uv_vis_experiment_names,
            self.photoluminescence_experiment_names,
            self.labels,
            self.colors,
        ):
            ax.plot(
                uv_vis_data[uv_vis_expt_name + "_wavelength"],  # x-axis
                uv_vis_data[uv_vis_expt_name + "_avg"],  # y-axis
                label=label,
                color=color,
            )
            ax2.plot(
                pl_data[pl_expt_name + "_wavelength"],  # x-axis
                pl_data[pl_expt_name + "_avg"],  # y-axis
                label=label,
                color=color,
                linestyle="--",  # https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
            )
            i += 1
        legend = ax.legend(
            loc="upper right",
            frameon=False,
            title="Sample IDs",
            fontsize=self.style["fontsize"],
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.add_artist(legend)

        plt.savefig(self.result_dir / f"{self.result_name}uv_vis_and_pl.png", dpi=300)
        plt.savefig(
            self.result_dir / f"{self.result_name}uv_vis_and_pl.eps",
            format="eps",
            dpi=300,
        )

    def plot_uv_vis(
        self,
        drop_columns: list[int],
        normalize: bool,
        baseline: bool,
        xlim: tuple,
        ylim: tuple = (-0.1, 1),
    ):
        """
        Function that plots photophysical data.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.tight_layout(pad=5)
        # uv_vis plot
        ax.set_xlabel("Wavelength (nm)", fontsize=self.style["fontsize"])
        ax.set_ylabel("Normalized Absorbance (a.u.)", fontsize=self.style["fontsize"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(
            axis="both", which="major", labelsize=self.style["fontsize"], direction="in"
        )

        # Set custom y-axis ticks (0, 0.5, 1.0)
        ax.set_yticks([0, 0.5, 1.0])

        # Set custom x-axis ticks with intervals of 100
        x_start = int(xlim[0] // 100) * 100  # Round down to nearest 100
        x_end = int(xlim[1] // 100 + 1) * 100  # Round up to nearest 100
        x_ticks = range(x_start, x_end + 1, 100)
        ax.set_xticks(x_ticks)

        i = 0
        uv_vis_data, pl_data = self.preprocess(drop_columns, normalize, baseline, xlim)

        for uv_vis_expt_name, label, color in zip(
            self.uv_vis_experiment_names,
            self.labels,
            self.colors,
        ):
            ax.plot(
                uv_vis_data[uv_vis_expt_name + "_wavelength"],  # x-axis
                uv_vis_data[uv_vis_expt_name + "_avg"],  # y-axis
                label=label,
                color=color,
            )
            i += 1
        legend = ax.legend(
            loc="upper right",
            frameon=False,
            title="Sample IDs",
            fontsize=self.style["fontsize"],
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.add_artist(legend)

        plt.savefig(self.result_dir / f"{self.result_name}uv_vis.png", dpi=300)
        plt.savefig(
            self.result_dir / f"{self.result_name}uv_vis.eps", format="eps", dpi=300
        )

    def plot_pl(
        self,
        drop_columns: list[int],
        normalize: bool,
        baseline: bool,
        xlim: tuple,
        ylim: tuple = (-0.1, 1),
    ):
        """
        Function that plots photophysical data.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.tight_layout(pad=5)
        # pl plot
        ax.set_ylabel("Normalized PL Intensity (a.u.)", fontsize=self.style["fontsize"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(
            axis="both", which="major", labelsize=self.style["fontsize"], direction="in"
        )
        i = 0
        # TODO: photoluminescence
        uv_vis_data, pl_data = self.preprocess(drop_columns, normalize, baseline, xlim)

        for pl_expt_name, label, color in zip(
            self.photoluminescence_experiment_names,
            self.labels,
            self.colors,
        ):
            ax.plot(
                pl_data[pl_expt_name + "_wavelength"],  # x-axis
                pl_data[pl_expt_name + "_avg"],  # y-axis
                label=label,
                color=color,
                linestyle="--",  # https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
            )
            i += 1
        legend = ax.legend(
            loc="upper right",
            frameon=False,
            title="Sample IDs",
            fontsize=self.style["fontsize"],
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.add_artist(legend)

        plt.savefig(self.result_dir / f"{self.result_name}pl.png", dpi=300)
        plt.savefig(
            self.result_dir / f"{self.result_name}pl.eps", format="eps", dpi=300
        )
