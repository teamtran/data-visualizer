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
        fluorescence_data_path: Path,
        experiment_name: list[str],
        labels: list[str],
        colors: list[str],
        result_dir: Path,
        style_path: Path,
    ):
        """
        Initialize the class with the data path and the result directory.
        """
        self.data_dir = data_dir
        self.uv_vis_data_path = uv_vis_data_path
        self.experiment_name = experiment_name
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
        uv_vis_data_path: Path,
        drop_columns: list[int],
        normalize: bool,
        baseline: bool,
    ) -> pd.DataFrame:
        """Function that applies transformation to the dataframe which will make it ready for plotting. Note, this is specific to uv_vis."""
        # Get all index of columns
        temp_data = pd.read_excel(uv_vis_data_path)
        columns = range(0, len(temp_data.columns))
        # Drop columns
        use_columns = [x for x in columns if x not in drop_columns]
        uv_vis_data = pd.read_excel(uv_vis_data_path, usecols=use_columns, skiprows=[1])
        # Baseline correction
        if baseline:
            pass
        # Normalize data
        if normalize:
            pass
        # TODO: triplicates
        return uv_vis_data_path

    def plot_uv_vis(
        self,
        uv_vis_metadata: list[str],
        drop_columns: list[int],
        normalize: bool,
        baseline: bool,
        xlim: tuple,
        ylim: tuple = (-0.1, 1),
    ):
        """
        Function that plots uv_vis data.
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
        for uv_vis_file, label, color in zip(
            self.uv_vis_data_path, self.labels, self.colors
        ):
            uv_vis_data = self.preprocess(
                uv_vis_data, drop_columns, normalize, baseline
            )  # normalize data
            ax.plot(
                uv_vis_data[uv_vis_data.columns[0]],
                uv_vis_data[uv_vis_data.columns[1]],
                label=label,
                color=color,
            )
            i += 1
        legend = ax.legend(
            loc="upper right",
            frameon=False,
            title="Experiment Conditions",
            fontsize=8,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # add another legend for uv_vis data (Mn, Mw, PDI)
        legend_handles = ax.get_legend_handles_labels()[0]
        uv_vis_legend = (legend_handles, uv_vis_metadata)
        uv_vis_metadata_legend = ax.legend(
            uv_vis_legend,
            labels=uv_vis_metadata,
            loc="center right",
            frameon=False,
            title="uv_vis Data",
            fontsize=8,
        )
        ax.add_artist(legend)

        plt.savefig(self.result_dir / f"{self.result_name}uv_vis_plot.png", dpi=400)
