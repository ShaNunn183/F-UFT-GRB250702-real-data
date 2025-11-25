"""
download_gbm_data.py

Download a Fermi GBM FITS light-curve file and convert it into simple
NumPy arrays: time (seconds) and count_rate (counts / second).

This script is intentionally simple and copy-paste friendly.
It uses an environment variable GBM_FITS_URL so you can point to
any specific GBM light-curve FITS file hosted at HEASARC or Fermi.

Example usage (local):

    export GBM_FITS_URL="https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/2025/bn250702xxx/current/glg_lc_b1_bn250702xxx_v00.fit"
    python src/download_gbm_data.py

If GBM_FITS_URL is not set, the script will exit with instructions.

Output:

    data/gbm_lightcurve.npz  # contains arrays "time" and "rate"
"""

import os
import pathlib
import sys
import numpy as np
import requests
from astropy.io import fits

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

def download_fits(url: str, out_path: pathlib.Path) -> None:
    print(f"[download_gbm_data] Downloading FITS from:\n  {url}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"[download_gbm_data] Saved FITS to: {out_path}")

def extract_lightcurve(fits_path: pathlib.Path, out_npz: pathlib.Path) -> None:
    print(f"[download_gbm_data] Reading FITS: {fits_path}")
    with fits.open(fits_path) as hdul:
        # This part may need to be adjusted depending on the exact
        # GBM product (CSPEC, CTIME, TTE). For most GBM light-curve
        # products, the light curve is stored in the first extension.
        hdu = hdul[1]

        # Try some common column names
        colnames = [c.upper() for c in hdu.columns.names]
        print(f"[download_gbm_data] Columns in extension 1: {colnames}")

        # Time column: usually "TIME"
        if "TIME" in colnames:
            time = np.array(hdu.data["TIME"], dtype=float)
        else:
            raise RuntimeError("Could not find TIME column in FITS file.")

        # Counts or rate column: try a few options
        rate = None
        for candidate in ["RATE", "COUNTS", "COUNT", "CTS", "FLUX"]:
            if candidate in colnames:
                rate = np.array(hdu.data[candidate], dtype=float)
                print(f"[download_gbm_data] Using column: {candidate}")
                break

        if rate is None:
            raise RuntimeError(
                "Could not find a suitable count / rate column "
                "in FITS file. Update download_gbm_data.py to match this file."
            )

        # If the rate is multi-channel (2D), sum over channels
        if rate.ndim > 1:
            print("[download_gbm_data] Detected multi-channel data, summing over channels.")
            rate = rate.sum(axis=1)

        # Basic cleaning: remove NaN or infinite values
        mask = np.isfinite(time) & np.isfinite(rate)
        time = time[mask]
        rate = rate[mask]

        print(f"[download_gbm_data] Final light curve length: {len(time)} points")

    np.savez(out_npz, time=time, rate=rate)
    print(f"[download_gbm_data] Saved cleaned light curve: {out_npz}")

def main() -> None:
    url = os.environ.get("GBM_FITS_URL", "").strip()
    if not url:
        msg = (
            "[download_gbm_data] ERROR: GBM_FITS_URL is not set.\n\n"
            "Set it to the direct URL of a Fermi GBM FITS light-curve file.\n"
            "Example (bash):\n"
            '    export GBM_FITS_URL="https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/2025/bn250702xxx/current/glg_lc_b1_bn250702xxx_v00.fit"\n"
            "Then run:\n"
            "    python src/download_gbm_data.py\n"
        )
        print(msg)
        sys.exit(1)

    fits_path = DATA_DIR / "gbm_lightcurve.fits"
    out_npz = DATA_DIR / "gbm_lightcurve.npz"

    download_fits(url, fits_path)
    extract_lightcurve(fits_path, out_npz)

if __name__ == "__main__":
    main()
