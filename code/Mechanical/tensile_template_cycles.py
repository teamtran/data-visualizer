import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pdb
from sklearn.metrics import r2_score


def linear_regime(data: pd.DataFrame, threshold: float = 100, sampling_rate: int = 5):
    # calculate first derivative of Stress vs. Strain every sampling_rate rows
    # pdb.set_trace()
    first_derivative = np.gradient(
        data["Stress (MPa)"][::sampling_rate], data["Strain (%)"][::sampling_rate]
    )
    # calculate second derivative of Stress vs. Strain
    second_derivative = np.gradient(
        first_derivative, data["Strain (%)"][::sampling_rate]
    )
    # find the row indices at which the second derivative is 0 +/- threshold
    linear_regime_idx = np.where(
        (second_derivative < (threshold / 1000))
        & (second_derivative > -(threshold / 1000))
    )[0]
    # get index at which the linear_regime_idx does not have consecutive values
    linear_regime_idx = np.split(
        linear_regime_idx, np.where(np.diff(linear_regime_idx) != 1)[0] + 1
    )[0]
    # get the first and last index of the linear regime
    linear_regime_start = linear_regime_idx[0] * sampling_rate
    linear_regime_end = linear_regime_idx[-1] * sampling_rate
    # calculate a linear fit for the linear regime
    linear_fit = np.polyfit(
        data["Strain (%)"][linear_regime_start:linear_regime_end],
        data["Stress (MPa)"][linear_regime_start:linear_regime_end],
        1,
    )
    # calculate the correlation coefficient of the linear fit
    r2 = r2_score(
        data["Stress (MPa)"][linear_regime_start:linear_regime_end],
        np.polyval(
            linear_fit, data["Strain (%)"][linear_regime_start:linear_regime_end]
        ),
    )
    return (linear_fit, r2, linear_regime_start, linear_regime_end)


if __name__ == "__main__":
    # NOTE: prior to loading data, add user-defined columns (thickness, width, cross-sectional area, load cell etc.)
    # TODO: Change data_path to the path of the data file
    # 23-11-3PetG2,8.8kdiaminewdioxTest014Data
    # 2024_03_28 013.A.6 4k10percent 30minvac  Test003Data
    data_path: Path = Path(
        "Mechanical/data/2024_04_26_013_A_8_4k10p_rate .2mm s-1 cycle 10pstrain Test010Data.csv"
    )

    # Load data
    data: pd.DataFrame = pd.read_csv(data_path)

    # Preprocess data (calculate strain and stress)
    # Find row index when Cycle is 1-Stretch
    start_idx = data.index[data["Cycle"] == "1-Stretch"][0]
    initial_size = data["Size_mm"][start_idx]

    # Calculating Strain
    data["Strain (%)"] = data["Displacement_mm"] * 100 / initial_size
    data["Stress (MPa)"] = data["Force_N"] / data["cross sectional area"][0]
    # Find row index when stress is near 0
    # error in stress calculation
    stress_error = data["load cell"][0] * 0.02 / data["cross sectional area"][0]
    # find the row index at which the strain is maximum
    stress_maximum = data["Stress (MPa)"].max()
    stress_maximum_index = data.index[data["Stress (MPa)"] == stress_maximum][0]
    end_idx = data[stress_maximum_index:].index[
        data["Stress (MPa)"][stress_maximum_index:] < stress_error
    ][0]
    #find row index at which cycles start
    cycle1_start_index = data.index[data["Cycle"] == "1-Stretch"][0]
    cycle2_start_index = data.index[data["Cycle"] == "2-Stretch"][0]
    cycle3_start_index = data.index[data["Cycle"] == "3-Stretch"][0]
    cycle4_start_index = data.index[data["Cycle"] == "4-Stretch"][0]
    cycle5_start_index = data.index[data["Cycle"] == "5-Stretch"][0]
    cycle6_start_index = data.index[data["Cycle"] == "6-Stretch"][0]
    cycle7_start_index = data.index[data["Cycle"] == "7-Stretch"][0]
    cycle8_start_index = data.index[data["Cycle"] == "8-Stretch"][0]
    cycle9_start_index = data.index[data["Cycle"] == "9-Stretch"][0]
    cycle10_start_index = data.index[data["Cycle"] == "10-Stretch"][0]
    # calculate linear regime
    print(data.loc[start_idx + 1 : stress_maximum_index])
    linear_fit, r2, linear_regime_start, linear_regime_end = linear_regime(
        data.loc[start_idx + 1 : stress_maximum_index]
    )

    # Plot data
    fig, ax = plt.subplots()
    ax.scatter(
        data["Strain (%)"][cycle1_start_index : cycle2_start_index-1],
        data["Stress (MPa)"][cycle1_start_index : cycle2_start_index-1],
        label="Cycle 1", color="#bedaf7", s=5
    ) 
    ax.scatter(
        data["Strain (%)"][cycle2_start_index : cycle3_start_index-1],
        data["Stress (MPa)"][cycle2_start_index : cycle3_start_index-1],
        label="Cycle 2", color="#7ab3ef", s=5
    )
    ax.scatter(
        data["Strain (%)"][cycle3_start_index : cycle4_start_index-1],
        data["Stress (MPa)"][cycle3_start_index : cycle4_start_index-1],
        label="Cycle 3", color="#368ce7", s=5
    )
    ax.scatter(
        data["Strain (%)"][cycle4_start_index : cycle5_start_index-1],
        data["Stress (MPa)"][cycle4_start_index : cycle5_start_index-1],
        label="Cycle 4", color="#1666ba", s=5
    )
    ax.scatter(
        data["Strain (%)"][cycle10_start_index :],
        data["Stress (MPa)"][cycle10_start_index :],
        label="Cycle 10", color="#be29ec", s=5
    )
    # plot linear fit and statistics
    ax.plot(
        data["Strain (%)"][linear_regime_start:linear_regime_end],
        np.polyval(
            linear_fit, data["Strain (%)"][linear_regime_start:linear_regime_end]
        ),
        label=f"Linear Fit: y = {linear_fit[0]:.3f}x + {linear_fit[1]:.3f} \n R^2 = {r2:.4f}",
        color="red",
    )

    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title("4k cycling 10% 0.2mm s-1")
    ax.legend()
    plt.savefig("Mechanical/results/2024_04_26_013_A_8_4k10p_rate cycle 0.2mm s-1 Data.png")
