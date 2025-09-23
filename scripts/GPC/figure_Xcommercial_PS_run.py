from pathlib import Path
import sys
from matplotlib import pyplot as plt
import json

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))

from code.GPC.gpc_plots import GPCPlots

# Style path
style_path: Path = root_path / "style" / "style.json"
# read style file
style: dict = json.loads(style_path.read_text())
# defines font, fontsize, and axes linewidth of all of the following plots
plt.rcParams["font.family"] = style["fontfamily"]
plt.rcParams["font.size"] = style["fontsize"]
plt.rcParams["axes.linewidth"] = style["axes_linewidth"]

# Experiment Name
experiment_name: str = "PS_Commercial"

# Result path
result_path: Path = root_path / "results" / "GPC" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "GPC" / experiment_name

# Define all comparison pairs with their metadata
comparison_sets = [
    {
        "filenames": [
            "2025_08_15_PS_Styrofoam.txt",
            "2025_08_15_PS_Styrofoam_SCF3.txt",
        ],
        "labels": ["PS-Styrofoam", "PS-Styrofoam-SCF3"],
        "metadata": [
            "Mn=65.9k, Mw=135.2k, Mp=115.0k, D=2.05",
            "Mn=69.2k, Mw=154.0k, Mp=109.6k, D=2.22",
        ],
    },
    {
        "filenames": [
            "20250902_PS_Red_Solo_Cup.txt",
            "20250902_PS_20250821-P5EER3-V1+2.txt",
        ],
        "labels": ["PS-Red-Solo-Cup", "PS-Red-Solo-Cup-SCF3"],
        "metadata": [
            "Mn=108k, Mw=203k, Mp=198k, D=1.88",
            "Mn=114k, Mw=221k, Mp=203k, D=1.94",
        ],
    },
    {
        "filenames": [
            "20250902_PS_Container.txt",
            "20250902_PS_20250825-P5EER5-V1+2.txt",
        ],
        "labels": ["PS-Container", "PS-Container-SCF3"],
        "metadata": [
            "Mn=128k, Mw=268k, Mp=248k, D=2.09",
            "Mn=164k, Mw=341k, Mp=282k, D=2.08",
        ],
    },
    {
        "filenames": [
            "20250902_PS_Cup_Lid.txt",
            "20250902_PS_20250822-P5EER4-V3+4.txt",
        ],
        "labels": ["PS-Coffee-Lid", "PS-Coffee-Lid-SCF3"],
        "metadata": [
            "Mn=90.5k, Mw=174.3k, Mp=151.5k, D=1.93",
            "Mn=135.4k, Mw=229.4k, Mp=205.1k, D=1.69",
        ],
    },
    {
        "filenames": [
            "2025_09_23_PS_Petri_Dish.txt",
            "2025_09_23_PS_Petri_Dish_SCF3.txt",
        ],
        "labels": ["PS-Petri-Dish", "PS-Petri-Dish-SCF3"],
        "metadata": [
            "Mn=101k, Mw=183k, Mp=159k, D=1.81",
            "Mn=105k, Mw=205k, Mp=165k, D=1.82",
        ],
    },
    {
        "filenames": [
            "20250902_PS_Blue_Cup.txt",
            "20250902_PS_20250821-P5EER3-V3+4.txt",
        ],
        "labels": ["PS-Blue-Cup", "PS-Blue-Cup-SCF3"],
        "metadata": [
            "Mn=118k, Mw=218k, Mp=235k, D=1.95",
            "Mn=133k, Mw=266k, Mp=266k, D=2.00",
        ],
    },
]

if __name__ == "__main__":
    # Generate plots for all comparison sets
    for i, comparison in enumerate(comparison_sets, 1):
        print(
            f"Generating plot {i}/{len(comparison_sets)}: {comparison['labels'][0]} vs {comparison['labels'][1]}"
        )

        # Check if both files exist before proceeding
        file1_path = data_dir / comparison["filenames"][0]
        file2_path = data_dir / comparison["filenames"][1]

        if not file1_path.exists():
            print(f"  Skipping: {comparison['filenames'][0]} not found")
            continue
        if not file2_path.exists():
            print(f"  Skipping: {comparison['filenames'][1]} not found")
            continue

        gpc_plots = GPCPlots(
            data_dir=data_dir,
            gpc_data_path=comparison["filenames"],
            labels=comparison["labels"],
            colors=[
                "#000000",
                "#ffcc02",
            ],
            result_dir=result_path,
            style_path=style_path,
        )

        gpc_plots.plot_gpc(
            gpc_metadata=comparison["metadata"],
            xlim=(4.5, 8),
            ylim=(-0.1, 1.1),
            inset_xlim=None,
            rt=7.25,
        )

        print(f"  ✓ Completed: {comparison['labels'][0]} vs {comparison['labels'][1]}")

    print(f"\nAll comparison plots generated! Check {result_path} for results.")
