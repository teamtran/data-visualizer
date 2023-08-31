from pathlib import Path
import pandas as pd
import json
import numpy as np


class Spectrum:
    """
    Class that contains the attributes and methods for a spectrum
    """

    def __init__(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        type: str,
        experiment_name: str,
        metadata_path: Path = None,
    ):
        self.wavelengths: np.ndarray = wavelength
        self.intensity: np.ndarray = intensity
        self.type: str = type
        self.experiment_name: str = experiment_name
        self.metadata: dict = self.from_json_update_metadata(metadata_path)
        self.update_units()
        self.update_intensity_by_dilution()

    @classmethod
    def from_analest_uv_vis(
        cls,
        path: Path,
        experiment_name: str,
        experiment_code: str,
        metadata_path: Path,
        type: str,
    ):
        """Gather uv_vis data into a single dataframe."""
        # uv_vis Data (one .csv with all data)
        # read file into dataframe and skip first 5 rows
        # Handle the path such that it reads the experiment_code.csv as the path
        experiment_code = experiment_code + ".csv"
        file_df: pd.DataFrame = pd.read_csv(
            path.parent / experiment_code, skiprows=[0, 1, 2, 3, 4]
        )
        # iterate through all experiments
        experiment_uv_vis = experiment_name + " 1"
        # get column idx if experiment is in column name
        # print(experiment_uv_vis)
        column_idx: int = file_df.columns.get_loc(experiment_uv_vis)
        abs: np.ndarray = file_df.iloc[:, column_idx].to_numpy()
        wavelength: np.ndarray = file_df.iloc[:, column_idx - 1].to_numpy()
        return cls(wavelength, abs, "uv_vis", experiment_name, metadata_path)

    @classmethod
    def from_analest_fluorescence(
        cls,
        path: Path,
        experiment_name: str,
        experiment_code: str,
        metadata_path: Path,
        type: str,
    ):
        # read file into dataframe
        file_df: pd.DataFrame = pd.read_csv(path, skiprows=[0])
        pl: np.ndarray = file_df["INT"].to_numpy()
        wavelength: np.ndarray = file_df["nm"].to_numpy()
        return cls(wavelength, pl, "photoluminescence", experiment_name, metadata_path)

    @classmethod
    def from_optics_table(
        cls,
        path: Path,
        experiment_name: str,
        experiment_code: str,
        metadata_path: Path,
        type: str,
    ):
        # read file into dataframe
        file_df: dict = pd.read_pickle(path)
        if type == "uv_vis":
            wavelength: np.ndarray = file_df["absorption"]["absorbance"][0]
            intensity: np.ndarray = file_df["absorption"]["absorbance"][1]
        elif type == "photoluminescence":
            wavelength: np.ndarray = file_df["PL"]["energy"][0]
            intensity: np.ndarray = file_df["PL"]["energy"][1]
        return cls(wavelength, intensity, type, experiment_name, metadata_path)

    def from_json_update_metadata(self, json_path: Path):
        with open(json_path) as json_file:
            metadata: dict = json.load(json_file)
            return metadata[self.experiment_name]

    def update_intensity_by_dilution(self):
        if self.metadata is not None:
            if (
                self.metadata["volume of photocatalyst (mL)"] != 0
                and self.metadata["volume of quencher (mL)"] != 0
            ):
                dilution_factor: float = self.metadata[
                    "volume of photocatalyst (mL)"
                ] / (
                    self.metadata["volume of quencher (mL)"]
                    + self.metadata["volume of photocatalyst (mL)"]
                )
            else:
                dilution_factor: float = 1
            self.intensity = self.intensity / dilution_factor
            self.metadata["dilution_factor"] = dilution_factor
        else:
            raise ValueError("No metadata found for experiment")
        pass

    def update_units(self):
        unit_labels: dict = {
            "optics": {"uv_vis": "", "photoluminescence": "energy/nm/s"},
            "ANALEST": {"uv_vis": "", "photoluminescence": "photon count"},
        }
        self.unit_label = unit_labels[self.metadata["instrument"]][self.type]
