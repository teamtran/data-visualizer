# File for handling folder structure and file paths
from pathlib import Path
from .spectrum import Spectrum
import re

# NOTE: Every user has to change this path to their own Dropbox path
dropbox_path: Path = (
    Path.home()
    / "Aspuru-Guzik Lab Dropbox"
    / "Stanley Lo"
    / "Stanley Lo"
    / "Research"
    / "AAG"
    / "Photocatalysis x Polymers"
    / "Data"
)


def get_csv_path(experiment_name: str) -> Path:
    return dropbox_path / "experiments" / f"{experiment_name}.csv"


def get_json_path(experiment_name: str) -> Path:
    return dropbox_path / "experiments" / f"{experiment_name}.json"


def get_analest_uv_vis_path(experiment_name: str) -> Path:
    return dropbox_path / "ANALEST" / "uv_vis" / f"{experiment_name}.csv"


def get_analest_fluorescence_path(experiment_name: str) -> Path:
    fluorimeter_path: Path = (
        dropbox_path / "ANALEST" / "fluorescence" / f"{experiment_name}"
    )
    file_path = None
    for file in fluorimeter_path.iterdir():
        if file.suffix == ".CSV":
            file_path = file
    return file_path


def get_optics_table_path(experiment_name: str) -> Path:
    return dropbox_path / "optics" / f"{experiment_name}" / f"{experiment_name}.pkl"


def get_plot_path(spectra: list[Spectrum], experiment_code: str) -> Path:
    # get all experiment names from spectra
    experiment_names: list[str] = [s.experiment_name.split("-")[1] for s in spectra]
    experiment_ids: list[int] = []
    for name in experiment_names:
        id: str = int(re.findall(r"\d+", name)[0])
        experiment_ids.append(id)
    experiment_ids.sort()
    experiment_name: str = f"{experiment_code}{experiment_ids[0]}-{experiment_ids[-1]}"

    return dropbox_path / "results" / f"{experiment_name}.png"
