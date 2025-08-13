from pathlib import Path
import sys
from matplotlib import pyplot as plt
import json
from typing import List

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))

from code.CV.cv_plots import CVPlots

# Style path
style_path: Path = root_path / "style" / "style.json"
# read style file
style: dict = json.loads(style_path.read_text())
# defines font, fontsize, and axes linewidth of all of the following plots
plt.rcParams["font.family"] = style["fontfamily"]
plt.rcParams["font.size"] = style["fontsize"]
plt.rcParams["axes.linewidth"] = style["axes_linewidth"]

# TODO: Change experiment name and filenames
# CV Curves
# Experiment Name
experiment_name: str = "SM-022-A12"

# Result path
result_path: Path = root_path / "results" / "CV" / experiment_name

# Data directory path
data_dir: Path = root_path / "data" / "CV" / experiment_name

# Filenames for single scan rate CV
single_cv_filenames: List[str] = [
    "SM-022-A12-MACROCYCLE(100MV)-1.2T0-REDUCTIONONLY-RERUN_C02.txt",
    # Add more files for comparison if needed
    # "SM-022-A12-MACROCYCLE(50MV)-1.2T0-REDUCTIONONLY-RERUN_C02.txt",
    # "SM-022-A12-MACROCYCLE(200MV)-1.2T0-REDUCTIONONLY-RERUN_C02.txt",
]

# Filenames for scan rate comparison (if you have multiple scan rates)
scan_rate_cv_filenames: List[str] = [
    # "SM-022-A12-MACROCYCLE(25MV)-1.2T0-REDUCTIONONLY-RERUN_C02.txt",
    # "SM-022-A12-MACROCYCLE(50MV)-1.2T0-REDUCTIONONLY-RERUN_C02.txt",
    # "SM-022-A12-MACROCYCLE(100MV)-1.2T0-REDUCTIONONLY-RERUN_C02.txt",
    # "SM-022-A12-MACROCYCLE(200MV)-1.2T0-REDUCTIONONLY-RERUN_C02.txt",
]

# Corresponding scan rates (mV/s) - match the order of scan_rate_cv_filenames
scan_rates: List[float] = [
    # 25,
    # 50,
    # 100,
    # 200,
]

# Labels for the plots
single_cv_labels: List[str] = [
    "MACROCYCLE-100mV_s",
    # "MACROCYCLE-50mV_s",
    # "MACROCYCLE-200mV_s",
]

scan_rate_labels: List[str] = [
    # "25 mV/s",
    # "50 mV/s",
    # "100 mV/s",
    # "200 mV/s",
]

# Colors for the plots
colors: List[str] = ["#f5c92a", "#8286ff", "#ad8c09", "#4772ff", "#ff6b6b", "#4ecdc4"]

if __name__ == "__main__":

    # Create results directory if it doesn't exist
    result_path.mkdir(parents=True, exist_ok=True)

    # Single CV plot(s)
    if single_cv_filenames:
        single_cv_plots = CVPlots(
            data_dir=data_dir,
            cv_data_path=single_cv_filenames,
            labels=single_cv_labels,
            colors=colors[: len(single_cv_filenames)],
            result_dir=result_path,
            style_path=style_path,
        )

        # TODO: Adjust parameters based on your CV data
        single_cv_plots.plot_cv_single(
            scan_rate=100,  # mV/s - change based on your actual scan rate
            xlim=(-1.5, 0.2),  # Voltage range in V - adjust based on your data
            ylim=(-0.015, 0.01),  # Current range in mA - adjust based on your data
            voltage_range=None,  # Filter voltage range if needed: (-2.0, 0.0)
            current_range=None,  # Filter current range if needed: (-0.01, 0.01)
            show_peaks=False,  # Mark anodic and cathodic peaks
            normalize_current=False,  # Set to True to normalize by scan rate
            figsize=(8, 6),
        )

    # Scan rate comparison plots (if you have multiple scan rates)
    # if scan_rate_cv_filenames and scan_rates:
    #     scan_rate_cv_plots = CVPlots(
    #         data_dir=data_dir,
    #         cv_data_path=scan_rate_cv_filenames,
    #         labels=scan_rate_labels,
    #         colors=colors[: len(scan_rate_cv_filenames)],
    #         result_dir=result_path,
    #         style_path=style_path,
    #     )

    # # Plot scan rate comparison
    # scan_rate_cv_plots.plot_cv_comparison_scan_rates(
    #     scan_rates=scan_rates,
    #     xlim=(-2.5, 0.5),  # Voltage range in V
    #     ylim=None,  # Let it auto-scale or set specific range
    #     normalize_current=True,  # Normalize by sqrt(scan_rate) for comparison
    #     figsize=(10, 6),
    # )

    # # Randles-Sevcik analysis for anodic peak
    # scan_rate_cv_plots.plot_randles_sevcik(
    #     scan_rates=scan_rates,
    #     peak_type="anodic",  # or "cathodic"
    #     figsize=(8, 6),
    # )

    # # Randles-Sevcik analysis for cathodic peak
    # scan_rate_cv_plots.plot_randles_sevcik(
    #     scan_rates=scan_rates,
    #     peak_type="cathodic",
    #     figsize=(8, 6),
    # )

    print(f"CV plots generated and saved to: {result_path}")

    # Optional: Show plots
    # plt.show()
