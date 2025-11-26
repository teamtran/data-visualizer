from pathlib import Path
import sys
import json
from matplotlib import pyplot as plt

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))

from code.TGA.tga_plots import TGAPlots

# Style path
style_path: Path = root_path / "style" / "style.json"
style: dict = json.loads(style_path.read_text())
plt.rcParams["font.family"] = style["fontfamily"]
plt.rcParams["font.size"] = style["fontsize"]
plt.rcParams["axes.linewidth"] = style["axes_linewidth"]

# TODO: Change experiment name and filenames
# Experiment Name
experiment_name: str = "isothermal_comparison_example"

# Result path
result_path: Path = root_path / "results" / "TGA" / experiment_name
if not result_path.exists():
    result_path.mkdir(parents=True)

# Data directory path
data_dir: Path = root_path / "data" / "TGA" / experiment_name

# Isothermal TGA filenames (replace with your actual files)
isothermal_filenames: list = [
    "sample1_isothermal_300C.csv",
    "sample2_isothermal_300C.csv",
    "sample3_isothermal_300C.csv",
]

# MS data (optional, can use None)
isothermal_ms_filenames: list = [
    None,
    None,
    None,
]

# Labels for each sample
labels: list = [
    "PS Pristine",
    "PS-SCF3 1%",
    "PS-SCF3 3%",
]

# Colors for plotting
colors: list = [
    "#000000",  # Black for pristine
    "#F4BD14",  # Gold for 1%
    "#a50026",  # Red for 3%
]

if __name__ == "__main__":
    # Create TGAPlots instance
    isothermal_tga = TGAPlots(
        data_dir=data_dir,
        tga_data_path=isothermal_filenames,
        ms_data_path=isothermal_ms_filenames,
        labels=labels,
        colors=colors,
        result_dir=result_path,
        style_path=style_path,
    )

    # Compare degradation rates at multiple mass loss percentages
    print("\n" + "=" * 80)
    print("Comparing Isothermal Degradation Rates")
    print("=" * 80)

    results = isothermal_tga.compare_isothermal_degradation_rates(
        mass_loss_targets=[10, 20, 30, 40, 50, 60, 70, 80],  # Target mass loss %
        initial_correction_time=50,  # Time to start analysis (min)
        xlim=(0, 1200),  # Time range for plotting
        ylim=(0, 100),  # Mass % range for plotting
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(results.to_string())
    print("\n")
    print("=" * 80)
