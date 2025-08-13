from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.ticker import ScalarFormatter


class CVPlots:
    """
    Class that contains methods to plot CV (Cyclic Voltammetry) data.
    """

    def __init__(
        self,
        data_dir: Path,
        cv_data_path: list[Path],
        labels: list[str],
        colors: list[str],
        result_dir: Path,
        style_path: Optional[Path] = None,
    ):
        """
        Initialize the class with the data path and the result directory.
        """
        self.data_dir = data_dir
        self.cv_data_path = cv_data_path
        self.labels = labels
        self.colors = colors
        self.result_dir = result_dir
        self.style_path = style_path
        if style_path and style_path.exists():
            self.style = json.load(open(style_path))
        else:
            self.style = {}
        self.result_name = ""
        for label in labels:
            self.result_name += label + "_"

    def preprocess(
        self,
        cv_data: pd.DataFrame,
        voltage_range: Optional[tuple] = None,
        current_range: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """
        Function that applies transformation to the dataframe which will make it ready for plotting.
        Note, this is specific to CV data.
        """
        # Clean column names - remove any extra whitespace
        cv_data.columns = cv_data.columns.str.strip()

        # Convert scientific notation strings to float if needed
        for col in cv_data.columns:
            if cv_data[col].dtype == "object":
                try:
                    cv_data[col] = pd.to_numeric(cv_data[col], errors="coerce")
                except:
                    pass

        # Filter data based on voltage range if specified
        if voltage_range is not None:
            mask = (cv_data.iloc[:, 1] >= voltage_range[0]) & (
                cv_data.iloc[:, 1] <= voltage_range[1]
            )
            cv_data = cv_data[mask]

        # Filter data based on current range if specified
        if current_range is not None:
            mask = (cv_data.iloc[:, 0] >= current_range[0]) & (
                cv_data.iloc[:, 0] <= current_range[1]
            )
            cv_data = cv_data[mask]

        return cv_data

    def get_peak_current(
        self, cv_data: pd.DataFrame, peak_type: str = "anodic"
    ) -> tuple:
        """
        Get the peak current and corresponding potential for the CV data.

        Args:
            cv_data: DataFrame containing CV data
            peak_type: "anodic" for positive current peak, "cathodic" for negative current peak

        Returns:
            tuple: (peak_current, peak_potential)
        """
        current_col = cv_data.columns[0]
        potential_col = cv_data.columns[1]

        if peak_type == "anodic":
            peak_idx = cv_data[current_col].idxmax()
        else:  # cathodic
            peak_idx = cv_data[current_col].idxmin()

        peak_current = cv_data.loc[peak_idx, current_col]
        peak_potential = cv_data.loc[peak_idx, potential_col]

        return peak_current, peak_potential

    def plot_cv_single(
        self,
        scan_rate: Optional[float] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        voltage_range: Optional[tuple] = None,
        current_range: Optional[tuple] = None,
        show_peaks: bool = False,
        normalize_current: bool = False,
        figsize: tuple = (8, 6),
    ):
        """
        Plot single or multiple CV curves for comparison.

        Args:
            scan_rate: Scan rate in mV/s for title
            xlim: x-axis limits (voltage)
            ylim: y-axis limits (current)
            voltage_range: Filter data to this voltage range
            current_range: Filter data to this current range
            show_peaks: Whether to mark anodic and cathodic peaks
            normalize_current: Whether to normalize current by scan rate
            figsize: Figure size
        """
        fig, ax = plt.subplots(1, figsize=figsize)

        # aesthetics
        ax.set_xlabel("Potential (V)", fontsize=12)
        if normalize_current and scan_rate:
            ax.set_ylabel("Current Density (mA/mV·s)", fontsize=12)
        else:
            ax.set_ylabel("Current (mA)", fontsize=12)

        if scan_rate:
            ax.set_title(f"Cyclic Voltammetry at {scan_rate} mV/s")
        else:
            ax.set_title("Cyclic Voltammetry")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")

        # Add zero current line
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.7)

        peak_data = []

        for cv_path, label, color in zip(self.cv_data_path, self.labels, self.colors):
            # Read CV data
            cv_data = pd.read_csv(
                self.data_dir / cv_path,
                sep="\t",  # assuming tab-separated based on your data
                encoding="utf-8",
                on_bad_lines="skip",
            )

            # Preprocess data
            cv_data = self.preprocess(cv_data, voltage_range, current_range)

            # Get column names (assuming first column is current, second is potential)
            current_col = cv_data.columns[0]
            potential_col = cv_data.columns[1]

            # Convert current from A to mA if needed (check units)
            current_data = cv_data[current_col]
            if (
                abs(current_data.max()) < 1e-3
            ):  # If max current is very small, likely in A
                current_data = current_data * 1000  # Convert A to mA

            # Normalize by scan rate if requested
            if normalize_current and scan_rate:
                current_data = current_data / scan_rate

            # Plot CV curve
            ax.plot(
                cv_data[potential_col],
                current_data,
                label=label,
                color=color,
                linewidth=2,
            )

            # Mark peaks if requested
            if show_peaks:
                try:
                    # Find anodic peak
                    anodic_current, anodic_potential = self.get_peak_current(
                        cv_data.assign(**{current_col: current_data}), "anodic"
                    )
                    ax.plot(
                        anodic_potential,
                        anodic_current,
                        "o",
                        color=color,
                        markersize=6,
                        markerfacecolor="none",
                        markeredgewidth=2,
                    )

                    # Find cathodic peak
                    cathodic_current, cathodic_potential = self.get_peak_current(
                        cv_data.assign(**{current_col: current_data}), "cathodic"
                    )
                    ax.plot(
                        cathodic_potential,
                        cathodic_current,
                        "s",
                        color=color,
                        markersize=6,
                        markerfacecolor="none",
                        markeredgewidth=2,
                    )

                    peak_data.append(
                        {
                            "label": label,
                            "anodic_peak": (anodic_potential, anodic_current),
                            "cathodic_peak": (cathodic_potential, cathodic_current),
                        }
                    )
                except:
                    print(f"Could not find peaks for {label}")

        # Set axis limits
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.legend()

        # Save plots
        scan_rate_str = f"_{scan_rate}mVs" if scan_rate else ""
        plt.savefig(
            self.result_dir / f"{self.result_name}cv{scan_rate_str}.png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.savefig(
            self.result_dir / f"{self.result_name}cv{scan_rate_str}.svg",
            dpi=400,
            bbox_inches="tight",
        )

        # Print peak information
        if show_peaks and peak_data:
            print("\nPeak Analysis:")
            for data in peak_data:
                print(f"{data['label']}:")
                print(
                    f"  Anodic peak: {data['anodic_peak'][1]:.3f} mA at {data['anodic_peak'][0]:.3f} V"
                )
                print(
                    f"  Cathodic peak: {data['cathodic_peak'][1]:.3f} mA at {data['cathodic_peak'][0]:.3f} V"
                )
                delta_e = data["anodic_peak"][0] - data["cathodic_peak"][0]
                print(f"  ΔE = {delta_e:.3f} V")

    def plot_cv_comparison_scan_rates(
        self,
        scan_rates: list[float],
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        normalize_current: bool = True,
        figsize: tuple = (10, 6),
    ):
        """
        Plot CV curves at different scan rates for comparison.

        Args:
            scan_rates: List of scan rates corresponding to each data file
            xlim: x-axis limits (voltage)
            ylim: y-axis limits (current)
            normalize_current: Whether to normalize current by sqrt(scan_rate)
            figsize: Figure size
        """
        if len(scan_rates) != len(self.cv_data_path):
            raise ValueError("Number of scan rates must match number of data files")

        fig, ax = plt.subplots(1, figsize=figsize)

        # aesthetics
        ax.set_xlabel("Potential (V)", fontsize=12)
        if normalize_current:
            ax.set_ylabel("Current / √(scan rate) (mA/(mV/s)^0.5)", fontsize=12)
        else:
            ax.set_ylabel("Current (mA)", fontsize=12)

        ax.set_title("Cyclic Voltammetry - Scan Rate Comparison")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")

        # Add zero current line
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.7)

        for cv_path, label, color, scan_rate in zip(
            self.cv_data_path, self.labels, self.colors, scan_rates
        ):
            # Read CV data
            cv_data = pd.read_csv(
                self.data_dir / cv_path, sep="\t", encoding="utf-8", on_bad_lines="skip"
            )

            # Preprocess data
            cv_data = self.preprocess(cv_data)

            # Get column names
            current_col = cv_data.columns[0]
            potential_col = cv_data.columns[1]

            # Convert current from A to mA if needed
            current_data = cv_data[current_col]
            if abs(current_data.max()) < 1e-3:
                current_data = current_data * 1000

            # Normalize by sqrt(scan_rate) if requested (for diffusion-controlled processes)
            if normalize_current:
                current_data = current_data / np.sqrt(scan_rate)

            # Plot CV curve
            ax.plot(
                cv_data[potential_col],
                current_data,
                label=f"{label} ({scan_rate} mV/s)",
                color=color,
                linewidth=2,
            )

        # Set axis limits
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.legend()

        # Save plots
        plt.savefig(
            self.result_dir / f"{self.result_name}cv_scan_rate_comparison.png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.savefig(
            self.result_dir / f"{self.result_name}cv_scan_rate_comparison.svg",
            dpi=400,
            bbox_inches="tight",
        )

    def plot_randles_sevcik(
        self,
        scan_rates: list[float],
        peak_type: str = "anodic",
        figsize: tuple = (8, 6),
    ):
        """
        Plot Randles-Sevcik analysis (peak current vs sqrt(scan rate)).

        Args:
            scan_rates: List of scan rates corresponding to each data file
            peak_type: "anodic" or "cathodic" peak to analyze
            figsize: Figure size
        """
        if len(scan_rates) != len(self.cv_data_path):
            raise ValueError("Number of scan rates must match number of data files")

        fig, ax = plt.subplots(1, figsize=figsize)

        sqrt_scan_rates = np.sqrt(scan_rates)
        peak_currents = []

        for cv_path, scan_rate in zip(self.cv_data_path, scan_rates):
            # Read and preprocess CV data
            cv_data = pd.read_csv(
                self.data_dir / cv_path, sep="\t", encoding="utf-8", on_bad_lines="skip"
            )
            cv_data = self.preprocess(cv_data)

            # Convert current from A to mA if needed
            current_col = cv_data.columns[0]
            if abs(cv_data[current_col].max()) < 1e-3:
                cv_data[current_col] = cv_data[current_col] * 1000

            # Get peak current
            peak_current, _ = self.get_peak_current(cv_data, peak_type)
            peak_currents.append(abs(peak_current))  # Use absolute value

        # Plot Randles-Sevcik
        ax.plot(sqrt_scan_rates, peak_currents, "o-", linewidth=2, markersize=8)

        # Linear fit
        coeffs = np.polyfit(sqrt_scan_rates, peak_currents, 1)
        fit_line = np.poly1d(coeffs)
        ax.plot(
            sqrt_scan_rates,
            fit_line(sqrt_scan_rates),
            "--",
            alpha=0.7,
            label=f"Linear fit: R² = {np.corrcoef(sqrt_scan_rates, peak_currents)[0,1]**2:.3f}",
        )

        ax.set_xlabel("√(Scan Rate) (mV/s)^0.5", fontsize=12)
        ax.set_ylabel(f"|{peak_type.capitalize()} Peak Current| (mA)", fontsize=12)
        ax.set_title("Randles-Sevcik Analysis")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
        ax.legend()

        # Save plot
        plt.savefig(
            self.result_dir / f"{self.result_name}randles_sevcik_{peak_type}.png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.savefig(
            self.result_dir / f"{self.result_name}randles_sevcik_{peak_type}.svg",
            dpi=400,
            bbox_inches="tight",
        )

        print(f"\nRandles-Sevcik Analysis ({peak_type} peak):")
        print(f"Slope: {coeffs[0]:.3f} mA/(mV/s)^0.5")
        print(f"Intercept: {coeffs[1]:.3f} mA")
        print(f"R²: {np.corrcoef(sqrt_scan_rates, peak_currents)[0,1]**2:.3f}")
