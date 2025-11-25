"""
analyze_gbm_fuft.py

Load a Fermi GBM light curve (time, rate) from data/gbm_lightcurve.npz
and test whether the power spectrum follows the F-UFT prediction:

    P(f) ~ f^{-(5 - D)}  with  D ~ 1.83  =>  slope ~ -3.17

Outputs:

- A log-log power spectrum plot saved as:
    results/grb250702_power_spectrum.png

- An estimated slope printed to stdout.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.stats import linregress

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def load_lightcurve(npz_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Could not find {npz_path}.\n"
            "Run download_gbm_data.py first to create the file."
        )
    data = np.load(npz_path)
    time = data["time"].astype(float)
    rate = data["rate"].astype(float)

    # Ensure sorted by time
    order = np.argsort(time)
    time = time[order]
    rate = rate[order]

    return time, rate

def compute_power_spectrum(time: np.ndarray, rate: np.ndarray):
    # Use uniform sampling: interpolate onto a regular grid
    t_min, t_max = time.min(), time.max()
    n_points = min(200000, len(time))  # cap to keep runtime reasonable
    t_uniform = np.linspace(t_min, t_max, n_points)
    rate_uniform = np.interp(t_uniform, time, rate)

    dt = t_uniform[1] - t_uniform[0]
    freqs = rfftfreq(len(t_uniform), d=dt)
    fft_vals = rfft(rate_uniform)
    power = np.abs(fft_vals) ** 2

    return freqs, power

def estimate_slope(freqs: np.ndarray, power: np.ndarray, f_min: float, f_max: float):
    """Estimate slope of log10(power) vs log10(freq) in a band."""
    band = (freqs > f_min) & (freqs < f_max)
    f_band = freqs[band]
    p_band = power[band]

    if len(f_band) < 10:
        raise RuntimeError("Not enough points in the selected frequency band.")

    log_f = np.log10(f_band)
    log_p = np.log10(p_band)

    slope, intercept, r_value, p_value, std_err = linregress(log_f, log_p)
    return slope, intercept, r_value, p_value, std_err

def main() -> None:
    npz_path = DATA_DIR / "gbm_lightcurve.npz"
    time, rate = load_lightcurve(npz_path)
    print(f"[analyze_gbm_fuft] Loaded light curve with {len(time)} points.")

    freqs, power = compute_power_spectrum(time, rate)
    print(f"[analyze_gbm_fuft] Computed power spectrum with {len(freqs)} frequency bins.")

    # Ignore zero frequency and very high frequencies dominated by noise.
    # Choose a band roughly in the middle of the usable range.
    f_min = max(freqs[1], 1.0 / (time.max() - time.min()) * 2.0)
    f_max = freqs[len(freqs) // 4]

    slope, intercept, r_value, p_value, std_err = estimate_slope(freqs, power, f_min, f_max)

    print("\n[F-UFT slope test]")
    print(f"  Estimated slope d log10(P) / d log10(f): {slope:.3f}")
    print(f"  Standard error: {std_err:.3f}")
    print(f"  Correlation r: {r_value:.3f}")
    print(f"  Expected F-UFT slope (D = 1.83): -3.17")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    # Use a subset to keep file size reasonable
    valid = (freqs > 0) & np.isfinite(power)
    ax.loglog(freqs[valid], power[valid], label="GBM data")

    # Overlay ideal F-UFT slope for visual guide
    # Choose a normalization near the middle of the band
    f_ref = 0.01
    p_ref = np.interp(f_ref, freqs[valid], power[valid])
    ideal = p_ref * (freqs[valid] / f_ref) ** (-3.17)
    ax.loglog(freqs[valid], ideal, "--", label="F-UFT f^-3.17")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (arb. units)")
    ax.set_title("GRB 250702BDE – Power Spectrum (Fermi GBM)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    out_png = RESULTS_DIR / "grb250702_power_spectrum.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"[analyze_gbm_fuft] Saved power spectrum plot to: {out_png}")

if __name__ == "__main__":
    main()
