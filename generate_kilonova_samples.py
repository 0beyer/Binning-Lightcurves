"""
Generate kilonova light-curve samples using the fiestaEM BullaFlux surrogate
(Bu2025_MLP, trained on POSSIS / Bulla 2025 radiative transfer grids).

Model: two-component ejecta — dynamical (neutron-rich, lanthanide-rich) and
       wind (electron-fraction-driven, bluer). The surrogate predicts apparent
       magnitudes in four optical/NIR bands.

Bands used (closest equivalents to g and r):
    bessellv   ~5500 Å  (g-proxy, equidistant from g 4770 Å and r 6230 Å)
    besselli   ~8060 Å  (r-proxy, next available band on red side)

Parameters sampled from prior (Bulla 2025 training range):
    inclination_EM   : viewing angle 0–90°  (sampled per source, not fixed)
    log10_mej_dyn    : dynamical ejecta mass  (log10 M_sun)
    v_ej_dyn         : dynamical ejecta velocity (c)
    Ye_dyn           : dynamical electron fraction
    log10_mej_wind   : wind ejecta mass  (log10 M_sun)
    v_ej_wind        : wind ejecta velocity (c)
    Ye_wind          : wind electron fraction

Distance:
    luminosity_distance sampled volume-weighted (uniform in d^3) over [d_min, d_max].
    Redshift derived from Hubble law (H0=70 km/s/Mpc), valid for d_L < ~500 Mpc.

Run (fiesta conda env):
    conda run -n fiesta python generate_kilonova_samples.py
    conda run -n fiesta python generate_kilonova_samples.py --n-samples 100
    conda run -n fiesta python generate_kilonova_samples.py --d-min 10 --d-max 500
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False

# fiesta must be imported before matplotlib.pyplot (it sets usetex=True globally)
from fiesta.inference.lightcurve_model import BullaFlux
from fiesta.inference.prior import Uniform, ConstrainedPrior
from fiesta.surrogates import download_surrogate

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "Bu2025_MLP"

# g- and r-band proxies (closest available in this surrogate)
FILTERS = ["bessellv", "besselli"]
BAND_COLORS = {
    "bessellv": "#2ca02c",   # green  (g-proxy, V ~5500 Å)
    "besselli": "#d62728",   # red    (r-proxy, I ~8060 Å)
}
BAND_LABELS = {
    "bessellv": "Bessell V ~5500 Å  (g-proxy)",
    "besselli": "Bessell I ~8060 Å  (r-proxy)",
}

H0 = 70.0       # km/s/Mpc  (Hubble constant for z approximation)
MAG_LIMIT = 30.0


# ── Prior ─────────────────────────────────────────────────────────────────────

def make_prior() -> ConstrainedPrior:
    return ConstrainedPrior([
        Uniform(xmin=0.,    xmax=np.pi/2,   naming=["inclination_EM"]),
        Uniform(xmin=-3.0,  xmax=-1.3,      naming=["log10_mej_dyn"]),
        Uniform(xmin=0.12,  xmax=0.28,      naming=["v_ej_dyn"]),
        Uniform(xmin=0.15,  xmax=0.35,      naming=["Ye_dyn"]),
        Uniform(xmin=-2.0,  xmax=-0.886,    naming=["log10_mej_wind"]),
        Uniform(xmin=0.05,  xmax=0.15,      naming=["v_ej_wind"]),
        Uniform(xmin=0.2,   xmax=0.4,       naming=["Ye_wind"]),
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--d-min", type=float, default=20.0,
                        help="Minimum luminosity distance in Mpc (default 20)")
    parser.add_argument("--d-max", type=float, default=400.0,
                        help="Maximum luminosity distance in Mpc (default 400)")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--output", default="kilonova_samples.png")
    parser.add_argument("--output-dir", default="kilonova_photometry",
                        help="Directory to dump per-sample CSV files (default: kilonova_photometry/)")
    args = parser.parse_args()

    # Load / download surrogate
    print(f"Loading surrogate '{MODEL_NAME}' ...")
    try:
        model = BullaFlux(name=MODEL_NAME, filters=FILTERS)
    except (OSError, ValueError):
        print("  Not found locally — downloading from Hugging Face ...")
        download_surrogate(MODEL_NAME)
        model = BullaFlux(name=MODEL_NAME, filters=FILTERS)

    print(f"  Time range: {model.times[[0, -1]]} days (source frame)")
    print(f"  Filters: {model.filters}")

    import os, csv
    import jax
    prior = make_prior()
    key   = jax.random.PRNGKey(args.seed)
    rng   = np.random.default_rng(args.seed)   # for distance sampling

    os.makedirs(args.output_dir, exist_ok=True)

    plt.rcParams["text.usetex"] = False   # fiesta re-enables usetex; override here
    fig, ax = plt.subplots(figsize=(11, 6))

    d_min3, d_max3 = args.d_min ** 3, args.d_max ** 3

    print(f"\nGenerating {args.n_samples} kilonova sample(s) "
          f"(iota random 0–90°, d_L {args.d_min}–{args.d_max} Mpc volume-weighted) ...")
    print(f"  Photometry CSVs → {args.output_dir}/")
    print(f"{'#':>3}  {'iota°':>6}  {'d_L':>7}  {'log_mdyn':>9}  {'v_dyn':>6}  "
          f"{'Ye_dyn':>6}  {'log_mwnd':>9}  {'v_wnd':>6}  {'Ye_wnd':>6}  v_peak")
    print("-" * 90)

    plotted = 0
    for i in range(args.n_samples):
        key, subkey = jax.random.split(key)
        draw   = prior.sample(subkey, 1)
        params = {k: float(v[0]) for k, v in draw.items()}

        # volume-weighted distance: uniform in d^3 → d = (u*(d_max^3-d_min^3)+d_min^3)^(1/3)
        d_L = float((rng.uniform() * (d_max3 - d_min3) + d_min3) ** (1.0 / 3.0))
        z   = H0 * d_L / 3e5          # Hubble-law approximation (valid < ~500 Mpc)
        params["luminosity_distance"] = d_L
        params["redshift"] = z

        times, mags = model.predict(params)
        times = np.array(times)

        # Report bessellv peak
        v_arr    = np.array(mags.get("bessellv", []))
        v_finite = v_arr[np.isfinite(v_arr)]
        v_peak   = v_finite.min() if len(v_finite) else np.nan

        iota_deg = np.degrees(params["inclination_EM"])
        print(f"{i+1:>3}  {iota_deg:>6.1f}  {d_L:>7.1f}  "
              f"{params['log10_mej_dyn']:>9.2f}  "
              f"{params['v_ej_dyn']:>6.3f}  {params['Ye_dyn']:>6.3f}  "
              f"{params['log10_mej_wind']:>9.2f}  "
              f"{params['v_ej_wind']:>6.3f}  {params['Ye_wind']:>6.3f}  "
              f"V={v_peak:.1f}")

        # Dump per-sample CSV: time | bessellv | besselli
        csv_path = os.path.join(args.output_dir, f"sample_{i+1:04d}.csv")
        mag_v = np.array(mags.get("bessellv", np.full_like(times, np.nan)))
        mag_i = np.array(mags.get("besselli", np.full_like(times, np.nan)))
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "# " + "  ".join(f"{k}={v:.4g}" for k, v in params.items()),
            ])
            writer.writerow(["time_days", "bessellv_mag_g-proxy", "besselli_mag_r-proxy"])
            for t, mv, mi in zip(times, mag_v, mag_i):
                writer.writerow([f"{t:.6f}", f"{mv:.6f}", f"{mi:.6f}"])

        alpha = 0.85 if args.n_samples == 1 else 0.35
        lw    = 2.2  if args.n_samples == 1 else 1.0
        # shade by distance: closer = more opaque
        dist_alpha = alpha * (1.0 - 0.5 * (d_L - args.d_min) / (args.d_max - args.d_min))

        for filt in FILTERS:
            mag  = np.array(mags.get(filt, []))
            good = np.isfinite(mag) & (mag < MAG_LIMIT + 2)
            if good.sum() < 2:
                continue
            ax.plot(times[good], mag[good],
                    color=BAND_COLORS[filt], lw=lw, alpha=dist_alpha)

        plotted += 1

    print(f"\nPlotted {plotted}/{args.n_samples} sample(s).")

    # Legend — band identity
    for filt in FILTERS:
        ax.plot([], [], color=BAND_COLORS[filt], lw=2, label=BAND_LABELS[filt])
    ax.legend(fontsize=10, loc="lower right")

    ax.invert_yaxis()
    ax.set_xlabel("Observer time (days post-merger)", fontsize=12)
    ax.set_ylabel("Apparent magnitude AB (bright at top)", fontsize=12)
    ax.set_title(
        f"Kilonova light curves — fiestaEM BullaFlux ({MODEL_NAME})\n"
        f"n={args.n_samples} sample(s)  |  iota random 0–90°  |  "
        f"d_L {args.d_min}–{args.d_max} Mpc (volume-weighted)  |  "
        f"bessellV (g-proxy) + bessellI (r-proxy)",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


# ── Fallback sampler (numpy-only, no JAX needed) ──────────────────────────────

def _numpy_sample(prior, rng):
    """Sample once from a ConstrainedPrior without JAX."""
    import jax, jax.numpy as jnp
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31)))
    return prior.sample(key, 1)


if __name__ == "__main__":
    main()
