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


class GPCPlots:
    """
    Class that contains methods to plot GPC data (multiple gpc curves altogether).
    """

    def __init__(
        self,
        data_dir: Path,
        gpc_data_path: list[Path],
        labels: list[str],
        colors: list[str],
        result_dir: Path,
        style_path: Path,
    ):
        """
        Initialize the class with the data path and the result directory.
        """
        self.data_dir = data_dir
        self.gpc_data_path = gpc_data_path
        self.labels = labels
        self.colors = colors
        self.result_dir = result_dir
        self.style_path = style_path
        self.style = json.load(open(style_path))
        self.result_name = ""
        for label in labels:
            self.result_name += label + "_"

    def preprocess(
        self, gpc_data: pd.DataFrame, normalize_space: np.array, i: int
    ) -> pd.DataFrame:
        """Function that applies transformation to the dataframe which will make it ready for plotting. Note, this is specific to GPC."""
        # Normalize data
        gpc_data[gpc_data.columns[1]] = (
            gpc_data[gpc_data.columns[1]] / gpc_data[gpc_data.columns[1]].max()
        ) * normalize_space[i]
        return gpc_data

    def plot_gpc(self, gpc_metadata: list[str], xlim: tuple, ylim: tuple = (-0.1, 1)):
        """
        Function that plots GPC data.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.tight_layout(pad=3)
        # aesthetics
        ax.set_xlabel("Retention Time (min)", fontsize=12)
        ax.set_ylabel("Normalized Intensity", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
        # Incrementally normalize the data to a lower point
        normalize_upper_limit = 1
        normalize_lower_limit = 0.5
        normalize_space = np.linspace(
            normalize_upper_limit, normalize_lower_limit, len(self.gpc_data_path)
        )
        i = 0
        for gpc_file, label, color in zip(self.gpc_data_path, self.labels, self.colors):
            gpc_data = pd.read_csv(self.data_dir / gpc_file, skiprows=1, sep="\t")
            gpc_data = self.preprocess(gpc_data, normalize_space, i)  # normalize data
            ax.plot(
                gpc_data[gpc_data.columns[0]],
                gpc_data[gpc_data.columns[1]],
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
        # add another legend for GPC data (Mn, Mw, PDI)
        legend_handles = ax.get_legend_handles_labels()[0]
        gpc_legend = (legend_handles, gpc_metadata)
        gpc_metadata_legend = ax.legend(
            gpc_legend,
            labels=gpc_metadata,
            loc="center right",
            frameon=False,
            title="GPC Data",
            fontsize=8,
        )
        ax.add_artist(legend)

        plt.savefig(self.result_dir / f"{self.result_name}gpc_plot.png", dpi=400)
