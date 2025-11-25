# F-UFT-GRB250702-real-data

Fractal-Unified Field Theory (F-UFT) analysis of GRB 250702BDE using real (or real-format) Fermi GBM light-curve data.

This repo contains:

- `src/download_gbm_data.py` – helper script to download a GBM FITS file and extract a light curve.
- `src/analyze_gbm_fuft.py` – power-spectrum analysis and F-UFT slope test.
- `.github/workflows/grb250702_fuft.yml` – GitHub Actions workflow to run the analysis on every push.
- `data/` – folder where the GBM FITS file and processed light curve will be stored.

The code is written to be:

- Minimal (only numpy, scipy, matplotlib, astropy, requests).
- Copy-paste friendly (ASCII math, no exotic symbols).
- Reproducible on a laptop, cluster, or GitHub Actions.
