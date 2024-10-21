from pathlib import Path


def get_filepath_auto(data_dir: Path) -> list[Path]:
    """Given a directory, if the directory has a data file, this function will return the path to the data file.

    Args:
        data_dir (Path): path to the directory with all the data

    Returns:
        list[Path]: filenames of the data files to plot
    """
    # Get all the datafiles in the directory and all subdirectories
    data_filenames = list(data_dir.rglob("*.xls"))
    return data_filenames
