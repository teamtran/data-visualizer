from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import json

# sets the path to the root of the repository
root_path = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_path))

# Style path
style_path: Path = root_path / "style" / "style.json"
# read style file
style: dict = json.loads(style_path.read_text())
# defines font, fontsize, and axes linewidth of all of the following plots
plt.rcParams["font.family"] = style["fontfamily"]
plt.rcParams["font.size"] = style["fontsize"]
plt.rcParams["axes.linewidth"] = style["axes_linewidth"]

# Result path
experiment_name: str = "arrhenius_analysis"
result_path: Path = root_path / "results" / "TGA" / experiment_name
if not result_path.exists():
    result_path.mkdir(parents=True)


def plot_arrhenius_comparison(
    delta_Ea: float = 8.0,
    Ea_base: float = 150.0,
    temp_range: tuple = (200, 500),
    A: float = 1e13,
):
    """
    Plot the effect of activation energy difference on rate constants.

    Args:
        delta_Ea: Difference in activation energy (kJ/mol)
        Ea_base: Base activation energy (kJ/mol)
        temp_range: Temperature range in Celsius (min, max)
        A: Pre-exponential factor (1/min or 1/s)
    """

    # Constants
    R = 8.314  # Gas constant in J/(mol·K)

    # Convert activation energies to J/mol
    Ea1 = Ea_base * 1000  # J/mol
    Ea2 = (Ea_base + delta_Ea) * 1000  # J/mol

    # Temperature range in Celsius
    T_celsius = np.linspace(temp_range[0], temp_range[1], 500)
    T_kelvin = T_celsius + 273.15

    # Calculate rate constants using Arrhenius equation: k = A * exp(-Ea/(RT))
    k1 = A * np.exp(-Ea1 / (R * T_kelvin))
    k2 = A * np.exp(-Ea2 / (R * T_kelvin))

    # Calculate ratio k1/k2 (lower Ea / higher Ea)
    k_ratio = k1 / k2

    # Calculate percent difference: (k1 - k2) / k2 * 100
    percent_diff = (k1 - k2) / k2 * 100

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # ========== Plot 1: Rate constants vs Temperature ==========
    ax1.semilogy(
        T_celsius,
        k1,
        linewidth=2.5,
        color="#2166ac",
        label=f"$E_a$ = {Ea_base:.0f} kJ/mol",
    )
    ax1.semilogy(
        T_celsius,
        k2,
        linewidth=2.5,
        color="#a50026",
        label=f"$E_a$ = {Ea_base + delta_Ea:.0f} kJ/mol",
    )

    ax1.set_xlabel("Temperature (°C)", fontsize=10)
    ax1.set_ylabel("Rate constant, k (min$^{-1}$)", fontsize=10)
    ax1.set_title(f"Arrhenius Equation: ΔE$_a$ = {delta_Ea:.1f} kJ/mol", fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="both", which="major", labelsize=8, direction="in")
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

    # ========== Plot 2: Ratio k1/k2 vs Temperature ==========
    ax2.plot(T_celsius, k_ratio, linewidth=2.5, color="#66bd63")
    ax2.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax2.set_xlabel("Temperature (°C)", fontsize=10)
    ax2.set_ylabel("Rate constant ratio (k$_1$/k$_2$)", fontsize=10)
    ax2.set_title(f"Rate Enhancement Factor", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="both", which="major", labelsize=8, direction="in")
    ax2.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

    # Add annotation for specific temperatures
    for T_annot in [250, 300, 350, 400]:
        if temp_range[0] <= T_annot <= temp_range[1]:
            idx = np.argmin(np.abs(T_celsius - T_annot))
            ax2.scatter(T_celsius[idx], k_ratio[idx], color="#66bd63", s=60, zorder=5)
            ax2.annotate(
                f"{k_ratio[idx]:.1f}×",
                (T_celsius[idx], k_ratio[idx]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    # ========== Plot 3: Percent Difference vs Temperature ==========
    ax3.plot(T_celsius, percent_diff, linewidth=2.5, color="#F4BD14")
    ax3.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax3.set_xlabel("Temperature (°C)", fontsize=10)
    ax3.set_ylabel("Percent Difference (%)", fontsize=10)
    ax3.set_title("(k$_1$ - k$_2$) / k$_2$ × 100", fontsize=10)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.tick_params(axis="both", which="major", labelsize=8, direction="in")
    ax3.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    # Save figure
    plt.savefig(
        result_path / f"arrhenius_comparison_deltaEa_{delta_Ea:.0f}kJ_mol.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.savefig(
        result_path / f"arrhenius_comparison_deltaEa_{delta_Ea:.0f}kJ_mol.eps",
        format="eps",
        dpi=400,
        bbox_inches="tight",
    )

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Arrhenius Rate Constant Analysis")
    print("=" * 80)
    print(f"Base Activation Energy (Ea1): {Ea_base:.1f} kJ/mol")
    print(f"Modified Activation Energy (Ea2): {Ea_base + delta_Ea:.1f} kJ/mol")
    print(f"Difference (ΔEa): {delta_Ea:.1f} kJ/mol")
    print(f"Pre-exponential Factor (A): {A:.2e} min^-1")
    print(f"\nTemperature Range: {temp_range[0]:.0f} - {temp_range[1]:.0f} °C")
    print("\n" + "-" * 80)
    print("Rate Constant Ratios (k1/k2) at Selected Temperatures:")
    print("-" * 80)

    for T in [250, 300, 350, 400, 450]:
        if temp_range[0] <= T <= temp_range[1]:
            idx = np.argmin(np.abs(T_celsius - T))
            print(
                f"  {T}°C ({T + 273.15:.0f}K): "
                f"k1 = {k1[idx]:.2e}, k2 = {k2[idx]:.2e}, "
                f"Ratio = {k_ratio[idx]:.2f}× "
                f"({percent_diff[idx]:.1f}% faster)"
            )

    print("\n" + "=" * 80)
    print(
        f"At lower temperatures, the difference in Ea has a LARGER effect on rate constant."
    )
    print(
        f"At higher temperatures, the difference in Ea has a SMALLER effect on rate constant."
    )
    print("=" * 80)

    plt.show()


def plot_arrhenius_linearized(
    delta_Ea: float = 8.0,
    Ea_base: float = 150.0,
    temp_range: tuple = (200, 500),
    A: float = 1e13,
):
    """
    Plot linearized Arrhenius equation: ln(k) vs 1/T.

    Args:
        delta_Ea: Difference in activation energy (kJ/mol)
        Ea_base: Base activation energy (kJ/mol)
        temp_range: Temperature range in Celsius (min, max)
        A: Pre-exponential factor (1/min or 1/s)
    """

    # Constants
    R = 8.314  # Gas constant in J/(mol·K)

    # Convert activation energies to J/mol
    Ea1 = Ea_base * 1000  # J/mol
    Ea2 = (Ea_base + delta_Ea) * 1000  # J/mol

    # Temperature range
    T_celsius = np.linspace(temp_range[0], temp_range[1], 500)
    T_kelvin = T_celsius + 273.15
    inv_T = 1000 / T_kelvin  # 1000/T for better scale

    # Calculate rate constants
    k1 = A * np.exp(-Ea1 / (R * T_kelvin))
    k2 = A * np.exp(-Ea2 / (R * T_kelvin))

    # Linearized form: ln(k) = ln(A) - Ea/(RT)
    ln_k1 = np.log(k1)
    ln_k2 = np.log(k2)

    # Calculate slopes: slope = -Ea/R
    slope1 = -Ea1 / R
    slope2 = -Ea2 / R

    # Create figure
    fig, ax = plt.subplots(1, figsize=(6, 4.5))

    # Plot linearized Arrhenius
    ax.plot(
        inv_T,
        ln_k1,
        linewidth=2.5,
        color="#2166ac",
        label=f"$E_a$ = {Ea_base:.0f} kJ/mol\n(slope = {slope1/1000:.1f}K)",
    )
    ax.plot(
        inv_T,
        ln_k2,
        linewidth=2.5,
        color="#a50026",
        label=f"$E_a$ = {Ea_base + delta_Ea:.0f} kJ/mol\n(slope = {slope2/1000:.1f}K)",
    )

    ax.set_xlabel("1000/T (K$^{-1}$)", fontsize=10)
    ax.set_ylabel("ln(k)", fontsize=10)
    ax.set_title(
        f"Linearized Arrhenius Plot: ΔE$_a$ = {delta_Ea:.1f} kJ/mol", fontsize=10
    )

    # Add secondary x-axis for temperature
    ax_top = ax.secondary_xaxis(
        "top",
        functions=(lambda x: 1000.0 / x - 273.15, lambda x: 1000.0 / (x + 273.15)),
    )
    ax_top.set_xlabel("Temperature (°C)", fontsize=10)
    ax_top.tick_params(axis="x", which="major", labelsize=8, direction="in")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=8, direction="in")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    # Save figure
    plt.savefig(
        result_path / f"arrhenius_linearized_deltaEa_{delta_Ea:.0f}kJ_mol.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.savefig(
        result_path / f"arrhenius_linearized_deltaEa_{delta_Ea:.0f}kJ_mol.eps",
        format="eps",
        dpi=400,
        bbox_inches="tight",
    )

    print(f"\nLinearized Arrhenius plots saved to {result_path}")

    plt.show()


if __name__ == "__main__":
    # Parameters
    delta_Ea = 4.6  # kJ/mol - difference in activation energy
    Ea_base = 300.0  # kJ/mol - base activation energy (typical for polymer degradation)
    temp_range = (200, 500)  # °C - temperature range
    A = 1e13  # min^-1 - pre-exponential factor (typical for degradation reactions)

    # Plot 1: Rate constant comparison
    print("\nGenerating Arrhenius comparison plots...")
    plot_arrhenius_comparison(
        delta_Ea=delta_Ea, Ea_base=Ea_base, temp_range=temp_range, A=A
    )

    # Plot 2: Linearized Arrhenius
    print("\nGenerating linearized Arrhenius plot...")
    plot_arrhenius_linearized(
        delta_Ea=delta_Ea, Ea_base=Ea_base, temp_range=temp_range, A=A
    )

    print(f"\nAll plots saved to: {result_path}")
