# First-Order Reaction Fitting Guide

This document explains the mathematical model and curve fitting approach used for analyzing TGA isothermal data in `plot_tga_isothermal_rate_constant()`.

## Mathematical Model

### First-Order Reaction Equation

For thermal degradation following first-order kinetics, mass percentage follows:

```
m(t) = m_∞ + (m_0 - m_∞) * exp(-k*t)
```

**Where:**
- `m(t)` = mass percentage at time t
- `m_0` = initial mass percentage (at t=0)
- `m_∞` = final/asymptotic mass percentage (as t→∞)
- `k` = rate constant (units: min⁻¹)
- `t` = time (min)

### Physical Interpretation

- **Rate constant (k)**: How quickly the reaction proceeds
  - Higher k → faster degradation
  - Units: min⁻¹ (inverse time)
  - Related to half-life: t₁/₂ = ln(2)/k

- **Total mass loss (m_0 - m_∞)**: Maximum degradation extent
  - Represents the degradable fraction of the sample
  - Units: % (percentage points)

- **Effective degradation rate**: k × (m_0 - m_∞)
  - Combines rate and extent of degradation
  - Better represents visual "steepness" of curves
  - Samples with different k values can appear similar if total mass loss differs

## Curve Fitting Implementation

### `scipy.optimize.curve_fit` Function

The function performs non-linear least squares optimization to find the best-fit parameters.

```python
popt, pcov = curve_fit(
    first_order_model,
    time_fit,
    mass_fit,
    p0=[m_inf_guess, m_0_guess, k_guess],
    maxfev=10000,
    bounds=(
        [0, 95, 0],    # Lower bounds: m_inf>=0, m_0>=95, k>=0
        [100, 100, 1], # Upper bounds: m_inf<=100, m_0<=100, k<=1
    ),
)
```

### Parameter Explanations

#### 1. `first_order_model` (function to fit)
- The mathematical model defined above
- First argument (t) is the independent variable
- Remaining arguments (m_inf, m_0, k) are parameters to optimize

#### 2. `time_fit` (x data)
- Array of time points (independent variable)
- Filtered to include only the desired fitting range (e.g., 0-1200 min)

#### 3. `mass_fit` (y data)
- Array of mass percentage values (dependent variable)
- Corresponds to each time point in `time_fit`

#### 4. `p0=[m_inf_guess, m_0_guess, k_guess]` (initial guesses)
**Purpose:** Starting point for the optimization algorithm

**Values:**
- `m_inf_guess`: Last data point in fitting range (~final mass)
- `m_0_guess`: First data point in fitting range (~initial mass)
- `k_guess`: 0.001 min⁻¹ (reasonable starting value)

**Why it matters:**
- Good guesses help optimizer converge faster
- Poor guesses can lead to local minima or convergence failure
- Order must match function parameter order: [m_inf, m_0, k]

#### 5. `maxfev=10000` (maximum function evaluations)
**Purpose:** Limit on optimization iterations

**Value:** 10,000 function calls
- Default is ~400 for 3 parameters
- Higher value allows more attempts to find optimal solution
- Prevents infinite loops if convergence is difficult

#### 6. `bounds=([0, 95, 0], [100, 100, 1])` (parameter constraints)
**Purpose:** Enforce physically realistic parameter ranges

**Lower bounds [0, 95, 0]:**
- `m_inf >= 0`: Final mass cannot be negative
- `m_0 >= 95`: Initial mass must be at least 95%
  - Data is normalized to start near 100%
  - Tighter bound prevents unrealistic fits
- `k >= 0`: Rate constant must be non-negative

**Upper bounds [100, 100, 1]:**
- `m_inf <= 100`: Final mass cannot exceed 100%
- `m_0 <= 100`: Initial mass cannot exceed 100%
- `k <= 1`: Rate constant capped at 1 min⁻¹
  - Prevents fitting extremely fast (unrealistic) degradation

**Why bounds matter:**
- Without bounds, optimizer might find mathematically "better" but physically meaningless solutions
- Example: m_0 = 80% with high k could fit data but contradicts experimental setup

### Return Values

#### `popt` (optimal parameters)
- Array of best-fit values: `[m_inf_fit, m_0_fit, k_fit]`
- These minimize the sum of squared residuals:
  ```
  Minimize: Σ(mass_fit - first_order_model(time_fit, m_inf, m_0, k))²
  ```

#### `pcov` (parameter covariance matrix)
- 3×3 matrix describing parameter uncertainties and correlations

**Structure:**
```
     m_inf      m_0        k
m_inf [[var_1,   cov_12,   cov_13],
m_0    [cov_21,  var_2,    cov_23],
k      [cov_31,  cov_32,   var_3]]
```

**Extracting uncertainties:**
```python
perr = np.sqrt(np.diag(pcov))
# perr = [std_error_m_inf, std_error_m_0, std_error_k]
```

**Diagonal elements:** Variances of each parameter
**Off-diagonal elements:** Covariances between parameter pairs

## Interpreting Results

### Key Metrics Reported

1. **Rate constant (k)**: Primary kinetic parameter
   - Compare between samples to assess relative degradation rates
   - Report with uncertainty: k ± σ_k

2. **Total mass loss (m_0 - m_∞)**: Extent of degradation
   - Higher values indicate more complete degradation
   - Different from rate constant!

3. **Effective degradation rate**: k × (m_0 - m_∞)
   - Combines rate and extent
   - Better for comparing visual "steepness"
   - Two samples with same effective rate can have very different k values

4. **Half-life**: ln(2)/k = 0.693/k
   - Time for mass to decay to midpoint between m_0 and m_∞
   - More intuitive than k for non-specialists

5. **R² score**: Goodness of fit
   - R² > 0.95: Excellent fit (first-order model appropriate)
   - R² < 0.90: Poor fit (consider different model or check data quality)

### Common Pitfalls

**Pitfall 1: Comparing only k values**
- Sample A: k=0.002, loses 40% → looks gradual
- Sample B: k=0.001, loses 80% → looks steep
- Sample A has higher k but appears slower!
- **Solution:** Compare effective degradation rates

**Pitfall 2: Ignoring uncertainties**
- Small differences in k may not be significant
- Check if error bars overlap before claiming differences
- **Solution:** Always report k ± σ_k

**Pitfall 3: Over-fitting noise**
- Very high R² (>0.999) with unrealistic parameters
- Usually from fitting too small a time range
- **Solution:** Use full experimental time range when possible

**Pitfall 4: Poor initial guesses**
- Optimizer gets stuck in local minimum
- Fitting fails or gives nonsensical results
- **Solution:** Inspect data first to set reasonable p0 values

## Example Output Interpretation

```
PS-10K:
  Rate constant (k): 1.234e-03 ± 5.67e-05 min^-1
  Initial mass (m_0): 98.50 ± 0.12%
  Final mass (m_∞): 35.20 ± 0.45%
  Total mass loss (m_0 - m_∞): 63.30%
  Effective degradation rate: 7.81e-02
  R² score: 0.9876
  Half-life: 561.64 min

PS-10K-SCF3:
  Rate constant (k): 9.876e-04 ± 4.23e-05 min^-1
  Initial mass (m_0): 99.10 ± 0.10%
  Final mass (m_∞): 28.90 ± 0.38%
  Total mass loss (m_0 - m_∞): 70.20%
  Effective degradation rate: 6.93e-02
  R² score: 0.9912
  Half-life: 701.67 min
```

**Interpretation:**
- PS-10K has higher k (1.23e-3 vs 9.88e-4), so degrades faster per first-order kinetics
- PS-10K-SCF3 has larger total mass loss (70.2% vs 63.3%), so more material degrades
- PS-10K has higher effective degradation rate (7.81e-2 vs 6.93e-2), so curve appears steeper
- Both fits are excellent (R² > 0.98)
- Half-lives differ by ~140 minutes, but uncertainties should be calculated to assess significance

## References

- Vyazovkin, S., et al. "ICTAC Kinetics Committee recommendations for performing kinetic computations on thermal analysis data." *Thermochimica Acta* 520.1-2 (2011): 1-19.
- Scipy documentation: [scipy.optimize.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
