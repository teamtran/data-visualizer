# Import all packages needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import json
from pathlib import Path
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# Import style.json
style_path: Path = Path.cwd() / "style" / "style.json"
with open(style_path, "r") as f:  # opens the style.json file
    style: dict = json.load(f)  # loads as a dictionary object

# Loading Data (data file must be in the same directory as jupyter notebook)
# Import data from excel file
# TODO: replace filename with the name of your data file
raw_data_filename: str = (
    Path.cwd() / "templates" / "2024_03_14_005a-F1and4Stan_data.xlsx"
)

# Tell program to read the data
raw_data: pd.DataFrame = pd.read_excel(
    raw_data_filename, skiprows=[1, 2]
)  # read data into a pandas dataframe and skip the first row since it is unnecessary


def preprocess(data: pd.DataFrame, normalize: bool = False):
    """Function that applies transformation to the dataframe which will make it ready for plotting. Note, this is specific to UV-Vis."""
    # Rename column names
    data = data.rename(
        columns={
            data.columns[0]: "Time (min)",
            data.columns[1]: "Refractive Index",
            data.columns[5]: "Time (min)",
            data.columns[6]: "Refractive Index",
        }
    )  # rename the first column to Time, and the second column to Refractive Index

    for i in range(0, 7):
        if "Time (min)" in data.columns[i]:
            # Find the index of the Time (min) column where the value is 45
            index: int = data[data.columns[i] == 45].index[
                0
            ]  # get the index of the row where the Time (min) column is equal to 45
            # Normalize absorbance values
            max_val: float = data[
                data.columns[i + 1]
            ].max()  # get the maximum value of the absorbance column
            data[data.columns[i + 1]] = (
                data[data.columns[i + 1]] / max_val
            )  # divide all values in the absorbance column by the maximum value to normalize the data
    return data


preprocessed_data = preprocess(raw_data)
print(preprocessed_data)
