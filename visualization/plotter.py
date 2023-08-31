import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from data.spectrum import Spectrum
from sklearn.metrics import r2_score
import math


# create a function to plot many individual Spectrum plots but with uv_vis and Photoluminescence Spectra stacked
def spectrum_subplots(spectrums: list[Spectrum], plot_path: Path):
    # Create a dictionary to store the spectrums based on experiment name
    grouped_spectrums = {}
    for spectrum in spectrums:
        experiment_name = spectrum.experiment_name
        if experiment_name in grouped_spectrums:
            grouped_spectrums[experiment_name].append(spectrum)
        else:
            grouped_spectrums[experiment_name] = [spectrum]
    # create a rectangular plot with several rows and columns
    rows: int = math.ceil(np.sqrt(len(grouped_spectrums)))
    cols: int = math.ceil(np.sqrt(len(grouped_spectrums)))
    # create figure and axes
    fig, ax = plt.subplots(cols, rows, figsize=(rows * 8, cols * 6))

    # Iterate through grouped spectrums
    for i, (experiment_name, spectra) in enumerate(grouped_spectrums.items()):
        # Calculate the subplot position
        subplot_pos = np.unravel_index(i, (cols, rows))

        # Create a twin axes for the second y-axis
        ax2 = ax[subplot_pos].twinx()

        # Iterate through the spectra within the experiment
        for spectrum in spectra:
            # Plot the spectrum in the corresponding subplot with different y-axis labels
            if (
                spectrum.type == "uv_vis"
            ):  # Replace 'spectrum_type' with the actual attribute name in your Spectrum class
                ax[subplot_pos].plot(
                    spectrum.wavelengths, spectrum.intensity, label=spectrum.type
                )
                ax[subplot_pos].set_ylabel(
                    f"uv_vis Intensity ({spectrum.unit_label})"
                )  # Set y-label for uv_vis spectrum
            elif (
                spectrum.type == "photoluminescence"
            ):  # Replace 'spectrum_type' with the actual attribute name in your Spectrum class
                ax2.plot(
                    spectrum.wavelengths,
                    spectrum.intensity,
                    label=spectrum.type,
                    color="orange",
                )
                ax2.set_ylabel(
                    f"Photoluminescence Intensity ({spectrum.unit_label})"
                )  # Set y-label for photoluminescence spectrum

        # Add subplot title as experiment name
        ax[subplot_pos].set_title(experiment_name)
        # Add legend to subplot
        ax[subplot_pos].legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax2.text(
            0.97,
            0.1,
            f'PC: {spectrum.metadata["photocatalyst"]}\
                \n solvent: {spectrum.metadata["solvent"]}\
                \n conc. of PC (M): {spectrum.metadata["concentration of photocatalyst (M)"]}\
                \n Q: {spectrum.metadata["quencher"]}\
                \n conc. of Q (M): {spectrum.metadata["concentration of quencher (M)"]}\
                \n dilution factor: {spectrum.metadata["dilution_factor"]}\
                \n atmosphere: {spectrum.metadata["atmosphere"]}\
                \n sparge_duration (s): {spectrum.metadata["sparge_duration (s)"]}\
                \n excitation wavelength (nm): {spectrum.metadata["excitation wavelength start (nm)"]}',
            fontsize=10,
            bbox=dict(facecolor="none", edgecolor="black", pad=5.0),
            ha="right",
            va="bottom",
            transform=ax2.transAxes,
        )
    # save figure
    plt.savefig(plot_path, dpi=300)


# create a function for plotting Spectrum but for a linear calibration curve
def calibration_curve(spectrums: list[Spectrum], lambda_max: int, plot_path: Path):
    # Only plots one type (absorption or photoluminescence) of spectrum
    # Create a dictionary to store the spectrums based on experiment name for grouping replicates
    grouped_spectrums = {}
    for spectrum in spectrums:
        replicate_name = spectrum.experiment_name.split("_")[0]
        if replicate_name in grouped_spectrums:
            grouped_spectrums[replicate_name].append(spectrum)
        else:
            grouped_spectrums[replicate_name] = [spectrum]
    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    # Iterate through grouped spectrums
    averaged_measurements: list[float] = []
    std_measurements: list[float] = []
    var1_conc_measurements: list[float] = []
    var2_conc_measurements: list[float] = []
    for i, (replicate_name, spectra) in enumerate(grouped_spectrums.items()):
        # get index of spectrum with lambda_max
        replicate_measurements: list[float] = []
        for spectrum in spectra:
            difference_array = abs(spectrum.wavelengths - lambda_max)
            spectrum_lambda_max_index: int = np.argmin(difference_array)
            replicate_measurements.append(spectrum.intensity[spectrum_lambda_max_index])
        averaged_measurements.append(np.mean(replicate_measurements))
        std_measurements.append(np.std(replicate_measurements))
        var1_conc_measurements.append(
            spectrum.metadata["concentration of photocatalyst (M)"]
        )
        var2_conc_measurements.append(
            spectrum.metadata["concentration of quencher (M)"]
        )
    # Plot the calibration curve
    # check if all numbers are the same in var1_conc_measurements
    if len(set(var1_conc_measurements)) != 1:
        conc = var1_conc_measurements
        label = "photocatalyst"
    else:
        conc = var2_conc_measurements
        label = "quencher"

    ax.scatter(conc, averaged_measurements, color="black", label=spectrum.type)
    ax.errorbar(conc, averaged_measurements, yerr=std_measurements, fmt="none")
    ax.set_xlabel(f"Concentration of {label} (M)")
    ax.set_ylabel(f"Intensity ({spectrum.unit_label})")
    # plot line of best fit
    # get slope and intercept
    slope, intercept = np.polyfit(conc, averaged_measurements, 1)
    # get x-values
    x_values: np.ndarray = np.array(conc)
    # get y-values
    y_values: np.ndarray = slope * x_values + intercept
    ax.plot(x_values, y_values, color="orange", label="Line of Best Fit")
    # plot equation
    ax.text(
        0.5, 0.5, f"y={round(slope, 3)}x+{round(intercept, 3)}", transform=ax.transAxes
    )
    # plot r^2 value
    ax.text(
        0.5,
        0.4,
        f"r={round(r2_score(averaged_measurements, y_values), 3)}",
        transform=ax.transAxes,
    )
    ax.set_title(
        f"{spectrum.type} \n vs. Concentration of Photocatalyst at {lambda_max} nm"
    )
    ax.legend(loc="best")
    # save figure
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")


# create a function for stacked spectrum plots
def stacked_spectra(spectrums: list[Spectrum], plot_path: Path):
    # Creates a stacked plot of all the spectra in the list for only one type (uv_vis or photoluminescence) of spectrum
    # Create a dictionary to store the spectrums based on experiment name for grouping replicates
    grouped_spectrums = {}
    for spectrum in spectrums:
        replicate_name = spectrum.experiment_name.split("_")[0]
        if replicate_name in grouped_spectrums:
            grouped_spectrums[replicate_name].append(spectrum)
        else:
            grouped_spectrums[replicate_name] = [spectrum]

    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    # TODO: implement scaling of color according to concentration (from dark green to light green)
    for i, (replicate_name, spectra) in enumerate(grouped_spectrums.items()):
        # get index of spectrum with lambda_max
        replicate_intensity: list[float] = []
        replicate_wavelength: list[float] = []
        for spectrum in spectra:
            replicate_intensity.append(spectrum.intensity)
            replicate_wavelength.append(spectrum.wavelengths)
        replicate_intensity = np.array(replicate_intensity)
        replicate_wavelength = np.array(replicate_wavelength)
        averaged_intensity = np.mean(replicate_intensity, axis=0)
        averaged_wavelength = np.mean(replicate_wavelength, axis=0)
        std_intensity = np.std(replicate_intensity, axis=0)
        std_wavelength = np.std(replicate_wavelength, axis=0)
        # Plot the calibration curve
        # Plot the spectra
        ax.plot(
            averaged_wavelength,
            averaged_intensity,
            color="black",
            label=f'[PC]: {spectrum.metadata["concentration of photocatalyst (M)"]} \
            \n [Q]: {spectrum.metadata["concentration of quencher (M)"]}',
        )
    ax.legend(loc="best")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"Intensity ({spectrum.unit_label})")
    ax.set_title(f"Stacked {spectrum.type} Spectra")
    # save figure
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
