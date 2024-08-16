from typing import Optional
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


class TGAPlots:
    """
    Class that contains methods to plot TGA-MS (dynamic, isothermal) data.
    """

    def __init__(
        self,
        data_dir: Path,
        tga_data_path: list[Path],
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
        self.tga_data_path = tga_data_path
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
        tga_data: pd.DataFrame,
        ms_data: pd.DataFrame,
        initial_correction: float,
        time_or_temp: str = "Time",
    ) -> pd.DataFrame:
        """Function that applies transformation to the dataframe which will make it ready for plotting. Note, this is specific to TGA-MS."""
        # Remove % sign from columns in tga_raw_data
        tga_data.columns = tga_data.columns.str.replace("%", "pct")
        # Truncate Temp./C column to Temp
        tga_data.columns = tga_data.columns.str.replace(tga_data.columns[0], "Temp")
        # Truncate Temp./C column to Temp in ms_raw_data
        ms_data.columns = ms_data.columns.str.replace(ms_data.columns[0], "Temp")
        # find the row closest to the initial_correction_time for tga_data
        if time_or_temp == "Time":
            initial_correction_row: int = tga_data.iloc[
                (tga_data["Time/min"] - initial_correction).abs().argsort()[:1]
            ].index[0]
            # find the row closest to the initial correction time for ms_data
            initial_correction_row_ms: int = ms_data.iloc[
                (ms_data["Time/min"] - initial_correction).abs().argsort()[:1]
            ].index[0]
        else:
            initial_correction_row: int = tga_data.iloc[
                (tga_data["Temp"] - initial_correction).abs().argsort()[:1]
            ].index[0]
            # find the row closest to the initial correction time for ms_data
            initial_correction_row_ms: int = ms_data.iloc[
                (ms_data["Temp"] - initial_correction).abs().argsort()[:1]
            ].index[0]
        # Subtract 100 from the Mass/% datapoint from the initial_correction_time_row
        correction_mass = 100 - tga_data["Mass/pct"][initial_correction_row]
        tga_data["Mass/pct"] = tga_data["Mass/pct"] + correction_mass
        # Remove the rows before the initial_correction_time_row for both tga_data and ms_data
        tga_data = tga_data.iloc[initial_correction_row:]
        ms_data = ms_data.iloc[initial_correction_row_ms:]
        return tga_data, ms_data

    def plot_tga_isothermal(
        self,
        isothermal_temp: float,
        target_mass: int = 104,
        xlim: tuple = (0, 1450),
        ylim: tuple = (0, 100),
        initial_correction_time: int = 50,
    ):
        """
        Plot several TGA isothermal data for comparison (can handle 1 or more).
        """
        fig, ax = plt.subplots(2, figsize=(6, 6))
        plt.subplots_adjust(hspace=0.5)
        # aesthetics
        ax[0].set_xlabel("Time (min)", fontsize=12)
        ax[0].set_ylabel("Mass (%)", fontsize=12)
        ax[0].set_title(f"Isothermal TGA at {isothermal_temp}°C")
        ax[1].set_xlabel("Time (min)", fontsize=12)
        ax[1].set_ylabel("Ion Current (A)", fontsize=12)
        ax[1].set_title(f"{target_mass} m/z for Isothermal TGA at {isothermal_temp}°C")
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[0].tick_params(axis="both", which="major", labelsize=12, direction="in")
        ax[1].tick_params(axis="both", which="major", labelsize=12, direction="in")
        # set xlim for the plots
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
        ax[0].set_ylim(ylim)
        ax[1].set_ylim(0, 1e-12)
        for tga_path, ms_path, label, color in zip(
            self.tga_data_path, self.ms_data_path, self.labels, self.colors
        ):
            tga_data = pd.read_csv(
                self.data_dir / tga_path,
                encoding="iso-8859-1",
                on_bad_lines="warn",
                skiprows=38,
            )
            ms_data = pd.read_csv(
                self.data_dir / ms_path,
                encoding="iso-8859-1",
                on_bad_lines="skip",
                skiprows=29,
            )
            tga_data, ms_data = self.preprocess(
                tga_data, ms_data, initial_correction_time, "Time"
            )
            ax[0].plot(
                tga_data["Time/min"], tga_data["Mass/pct"], label=label, color=color
            )
            ax[1].plot(
                ms_data["Time/min"],
                ms_data[f"QMID(s:1|m:{target_mass})/A"],
                label=label,
                color=color,
            )
        ax[0].legend()
        ax[1].legend()
        plt.savefig(
            self.result_dir
            / f"{self.result_name}tga_isothermal_{isothermal_temp}_{target_mass}m_z.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir
            / f"{self.result_name}tga_isothermal_{isothermal_temp}_{target_mass}m_z.svg",
            dpi=400,
        )

    def plot_tga_dynamic(
        self,
        target_mass: int = 104,
        t_depolymerization_cutoff: float = 99,
        xlim: tuple = (100, 400),
        initial_correction_temp: int = 230,
    ):
        # TODO: initial correction temp instead of time
        """
        Plot the TGA dynamic data.
        """
        fig, ax = plt.subplots(2, figsize=(6, 6))
        plt.subplots_adjust(hspace=0.5)
        # aesthetics
        ax[0].set_xlabel("Temp (oC)", fontsize=12)
        ax[0].set_ylabel("Mass (%)", fontsize=12)
        ax[0].set_title(f"Dynamic TGA")
        ax[1].set_xlabel("Temp (oC)", fontsize=12)
        ax[1].set_ylabel("Ion Current (A)", fontsize=12)
        ax[1].set_title(f"{target_mass} m/z for Dynamic TGA")
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[0].tick_params(axis="both", which="major", labelsize=12, direction="in")
        ax[1].tick_params(axis="both", which="major", labelsize=12, direction="in")
        for tga_path, ms_path, label, color in zip(
            self.tga_data_path, self.ms_data_path, self.labels, self.colors
        ):
            print(tga_path)
            tga_data = pd.read_csv(
                self.data_dir / tga_path,
                encoding="iso-8859-1",
                on_bad_lines="warn",
                skiprows=37,
            )
            ms_data = pd.read_csv(
                self.data_dir / ms_path,
                encoding="iso-8859-1",
                on_bad_lines="skip",
                skiprows=29,
            )
            tga_data, ms_data = self.preprocess(
                tga_data, ms_data, initial_correction_temp, "Temp"
            )
            ax[0].plot(tga_data["Temp"], tga_data["Mass/pct"], label=label, color=color)
            ax[1].plot(
                ms_data["Temp"],
                ms_data[f"QMID(s:1|m:{target_mass})/A"],
                label=label,
                color=color,
            )
        # Set colors for the axis labels
        # Draw a line in the y-axis at y=99%
        ax[0].axhline(
            y=t_depolymerization_cutoff,
            color="r",
            linestyle="--",
            linewidth=0.3,
            label="$T_\mathregular{depolymerization}$"
            + f" at {t_depolymerization_cutoff}% Mass",
        )
        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
        plt.savefig(
            self.result_dir / f"{self.result_name}tga_dynamic_{target_mass}m_z.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir / f"{self.result_name}tga_dynamic_{target_mass}m_z.svg",
            dpi=400,
        )

    def plot_ms_peak_area(
        self,
        tga_type: str,
        isothermal_temp: Optional[float],
        initial_time: float = 40,
        end_time: float = 1450,
        m_z_start: int = 60,
        m_z_end: int = 150,
        normalize: bool = True,
    ):
        """
        Plot the peak area for each m/z over time.
        x-axis: m/z
        y-axis: peak area from time 0 to time x.
        """
        fig, ax = plt.subplots(1, figsize=(8, 6))
        # aesthetics
        ax.set_xlabel("m/z", fontsize=12)
        ax.set_ylabel(
            "Normalized Peak Area of Ion Current over Time (A x min)", fontsize=12
        )
        if tga_type == "isothermal":
            ax.set_title(
                f"Normalized Peak Area for m/z {m_z_start} to {m_z_end} for {tga_type} TGA-MS data at {isothermal_temp}°C"
            )
        else:
            ax.set_title(
                f"Normalized Peak Area for m/z {m_z_start} to {m_z_end} for {tga_type} TGA-MS data"
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
        for ms_path, label, color in zip(self.ms_data_path, self.labels, self.colors):
            ms_data = pd.read_csv(
                self.data_dir / ms_path,
                encoding="iso-8859-1",
                on_bad_lines="skip",
                skiprows=29,
            )
            ms_data = ms_data.iloc[
                (ms_data["Time/min"] - initial_time)
                .abs()
                .argsort()[: (ms_data["Time/min"] - end_time).abs().argsort()[0]]
            ]
            # find index at which the m/z starts and ends
            m_z_start_idx: int = ms_data.columns.get_loc(f"QMID(s:1|m:{m_z_start})/A")
            m_z_end_idx: int = ms_data.columns.get_loc(f"QMID(s:1|m:{m_z_end})/A")
            peak_area = ms_data.iloc[:, m_z_start_idx:m_z_end_idx].sum(axis=0)
            if normalize:
                peak_area = peak_area / peak_area.max()
            ax.plot(
                range(m_z_start, m_z_end),
                peak_area,
                label=label,
                color=color,
                marker="o",
                markersize=2,
            )
        # label all the points with m/z that are above 10% of the max peak area
        for i, txt in enumerate(peak_area):
            if txt > 0.05 * peak_area.max():
                ax.annotate(
                    f"{range(m_z_start, m_z_end)[i]}",
                    (range(m_z_start, m_z_end)[i], peak_area[i]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                )

        ax.legend()
        plt.savefig(
            self.result_dir
            / f"{self.result_name}ms_peak_area_{m_z_start}_{m_z_end}_{isothermal_temp}.png",
            dpi=400,
        )
        plt.savefig(
            self.result_dir
            / f"{self.result_name}ms_peak_area_{m_z_start}_{m_z_end}_{isothermal_temp}.svg",
            dpi=400,
        )
