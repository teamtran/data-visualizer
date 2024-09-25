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
        uv_vis_data_path: Path,
        photoluminescence_data_path: Path,
        uv_vis_experiment_names: list[str],
        photoluminescence_experiment_names: list[str],
        labels: list[str],
        colors: list[str],
        result_dir: Path,
        style_path: Path,
    ):
        """
        Initialize the class with the data path and the result directory.
        """
        self.data_dir = data_dir
        self.uv_vis_data_path = data_dir / uv_vis_data_path
        self.photoluminescence_data_path = data_dir / photoluminescence_data_path
        self.uv_vis_experiment_names = uv_vis_experiment_names
        self.photoluminescence_experiment_names = photoluminescence_experiment_names
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
        drop_columns: list[int],
        normalize: bool,
        baseline: bool,
    ) -> pd.DataFrame:
        """Function that applies transformation to the dataframe which will make it ready for plotting. Note, this is specific to photophysical."""
        # TODO: Photoluminescence
        # Get all index of columns
        temp_data = pd.read_excel(self.uv_vis_data_path)
        columns = range(0, len(temp_data.columns))
        # Drop columns
        use_columns = [x for x in columns if x not in drop_columns]
        # Drop all rows after the data (i.e. drop the metadata)
        # Find row all NaNs
        row_to_drop = temp_data[temp_data.isnull().all(axis=1)].index[0]

        num_of_rows = len(temp_data.index)
        skiprows = list(range(row_to_drop, num_of_rows + 1))
        skiprows.append(1)  # skip row with wavelength and absorbance
        uv_vis_data = pd.read_excel(
            self.uv_vis_data_path, usecols=use_columns, skiprows=skiprows
        )
        # Baseline correction
        for expt in self.uv_vis_experiment_names:
            # get all column_idx with expt name
            column_idx = [
                i + 1 for i, column in enumerate(uv_vis_data.columns) if expt in column
            ]
            for col_idx in column_idx:
                if baseline:
                    absorbance = uv_vis_data.iloc[:, col_idx]
                    min_absorbance = min(absorbance)
                    uv_vis_data.iloc[:, col_idx] = absorbance - min_absorbance
                # Normalize data
                if normalize:  # divide by max absorbance
                    absorbance = uv_vis_data.iloc[:, col_idx]
                    max_absorbance = max(absorbance)
                    absorbance = absorbance / max_absorbance
            # average triplicates
            uv_vis_data[expt + "_wavelength"] = uv_vis_data.iloc[
                :, col_idx - 1
            ]  # wavelength from last replicate
            uv_vis_data[expt + "_avg"] = uv_vis_data.iloc[:, column_idx].mean(axis=1)
        return uv_vis_data

    def plot_photophysical(
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
        plt.tight_layout(pad=3)
        # aesthetics
        ax.set_xlabel("Wavelength (nm)", fontsize=12)
        ax.set_ylabel("Normalized Absorbance (a.u.)", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
        i = 0
        # TODO: photoluminescence
        uv_vis_data = self.preprocess(drop_columns, normalize, baseline)

        for expt_name, label, color in zip(
            self.uv_vis_experiment_names, self.labels, self.colors
        ):
            ax.plot(
                uv_vis_data[expt_name + "_wavelength"],  # x-axis
                uv_vis_data[expt_name + "_avg"],  # y-axis
                label=label,
                color=color,
            )
            i += 1
        legend = ax.legend(
            loc="upper right",
            frameon=False,
            title="Sample IDs",
            fontsize=8,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.add_artist(legend)

        plt.savefig(
            self.result_dir / f"{self.result_name}photophysical_plot.png", dpi=400
        )
