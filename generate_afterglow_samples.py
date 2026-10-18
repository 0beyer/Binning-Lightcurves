"""
Generate GRB afterglow light-curve samples using afterglowpy (Gaussian jet)
and plot apparent magnitude vs time for PS1 g and r bands.

Samples physical parameters from the same prior used in fiestaEM inference_GRB.py.
Each sample draws: viewing angle (sine prior 0–45°), jet energy, core/wing angles,
density, electron index, and microphysics fractions.

Distance is sampled volume-weighted (uniform in d^3) over [d_min, d_max].
Viewing angle is sampled from the prior per source (not fixed).

Data is streamed to Kafka for real-time binning via streaming_binner.py (default).
Use --no-kafka to disable Kafka streaming and only save to CSV files.
The --m-bins parameter controls bin size for both Kafka streaming and post-processing.

To simulate argus, a good number could be --n-samples 10000

Run (fiesta conda env):
    # Default: Real-time Kafka streaming with bin size 5
    conda run -n fiesta python generate_afterglow_samples.py --n-samples 100

    # Custom bin size for Kafka streaming
    conda run -n fiesta python generate_afterglow_samples.py --n-samples 100 --m-bins 5

    # Custom distance range
    conda run -n fiesta python generate_afterglow_samples.py --n-samples 100 --d-min 10 --d-max 500
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

import afterglowpy as grb
from kafka_producer_util import LightCurveProducer

# ── Physical constants / conversions ─────────────────────────────────────────
C_ANGSTROM  = 2.997_924_58e18   # speed of light in Å/s
MPC_TO_CM   = 3.085_677_58e24   # 1 Mpc in cm
DAY_TO_SEC  = 86400.0

# ── Band central frequencies (Hz) — g and r only ─────────────────────────────
BANDS = {
    "g (PS1, 4810 Å)": C_ANGSTROM / 4810.0,
    "r (PS1, 6170 Å)": C_ANGSTROM / 6170.0,
}
BAND_COLORS = {
    "g (PS1, 4810 Å)": "#2ca02c",
    "r (PS1, 6170 Å)": "#d62728",
}

H0 = 70.0   # km/s/Mpc

# ── Prior bounds (match fiestaEM inference_GRB.py) ───────────────────────────
PRIOR = {
    "thetaObs":    (0.0,  np.pi / 4),   # Sine prior; range 0–45 deg
    "log10_E0":    (47.0, 57.0),
    "thetaCore":   (0.01, np.pi / 5),
    "alphaWing":   (0.2,  3.5),          # thetaWing = thetaCore * alphaWing
    "log10_n0":    (-6.0, 2.0),
    "p":           (2.01, 3.0),
    "log10_eps_e": (-4.0, 0.0),
    "log10_eps_B": (-8.0, 0.0),
}

MAG_LIMIT = 30.0    # only plot curves that reach brighter than this


# ── Helpers ───────────────────────────────────────────────────────────────────

def sample_prior(rng: np.random.Generator) -> dict:
    """Draw one parameter set from the prior."""

    # Sine prior for inclination (uniform in cos(theta) → denser near 0)
    thetaObs_max = PRIOR["thetaObs"][1]
    thetaObs = np.arccos(rng.uniform(np.cos(thetaObs_max), 1.0))

    thetaCore = rng.uniform(*PRIOR["thetaCore"])
    alphaWing = rng.uniform(*PRIOR["alphaWing"])
    thetaWing = thetaCore * alphaWing
    # Constrain thetaWing < pi/2
    thetaWing = min(thetaWing, np.pi / 2 - 1e-4)

    log10_eps_e = rng.uniform(*PRIOR["log10_eps_e"])
    log10_eps_B = rng.uniform(*PRIOR["log10_eps_B"])
    # Constrain epsilon_e + epsilon_B < 1
    while 10**log10_eps_e + 10**log10_eps_B >= 1.0:
        log10_eps_e = rng.uniform(*PRIOR["log10_eps_e"])
        log10_eps_B = rng.uniform(*PRIOR["log10_eps_B"])

    return {
        "thetaObs":    thetaObs,
        "log10_E0":    rng.uniform(*PRIOR["log10_E0"]),
        "thetaCore":   thetaCore,
        "thetaWing":   thetaWing,
        "log10_n0":    rng.uniform(*PRIOR["log10_n0"]),
        "p":           rng.uniform(*PRIOR["p"]),
        "log10_eps_e": log10_eps_e,
        "log10_eps_B": log10_eps_B,
    }


def afterglow_mags(p: dict, t_days: np.ndarray, d_L_cm: float) -> dict[str, np.ndarray]:
    """
    Compute AB magnitudes for each band using afterglowpy (Gaussian jet).

    Returns dict {band_label: magnitude_array}, with np.nan where flux <= 0.
    """
    t_sec = t_days * DAY_TO_SEC

    Z = {
        "jetType":   grb.Gaussian,
        "specType":  0,
        "thetaObs":  p["thetaObs"],
        "E0":        10 ** p["log10_E0"],
        "thetaCore": p["thetaCore"],
        "thetaWing": p["thetaWing"],
        "n0":        10 ** p["log10_n0"],
        "p":         p["p"],
        "epsilon_e": 10 ** p["log10_eps_e"],
        "epsilon_B": 10 ** p["log10_eps_B"],
        "xi_N":      1.0,
        "d_L":       d_L_cm,
    }

    mags = {}
    for label, nu_hz in BANDS.items():
        nu_arr = np.full_like(t_sec, nu_hz)
        try:
            Fnu_mJy = np.asarray(grb.fluxDensity(t_sec, nu_arr, **Z),
                                 dtype=float)   # afterglowpy returns mJy
            Fnu_uJy = Fnu_mJy * 1e3            # mJy → μJy (ZP = 23.9)
            with np.errstate(divide="ignore", invalid="ignore"):
                mag = np.where(Fnu_uJy > 0,
                               23.9 - 2.5 * np.log10(Fnu_uJy),
                               np.nan)
        except Exception:
            mag = np.full_like(t_sec, np.nan)
        mags[label] = mag
    return mags

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import os, csv

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--d-min",     type=float, default=20.0,
                        help="Minimum luminosity distance in Mpc (default 20)")
    parser.add_argument("--d-max",     type=float, default=400.0,
                        help="Maximum luminosity distance in Mpc (default 400)")
    parser.add_argument("--m-bins", type=int, default=0,
                        help="Bin size for Kafka streaming, off by default)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    default="afterglow_samples.png")
    parser.add_argument("--output-dir", default="afterglow_photometry",
                        help="Directory to dump per-sample CSV files (default: afterglow_photometry/)")
    parser.add_argument("--no-kafka", action="store_true",
                        help="Disable Kafka streaming (only save to CSV)")
    parser.add_argument("--kafka-bootstrap-servers", default="localhost:9092",
                        help="Kafka bootstrap servers (default: localhost:9092)")
    parser.add_argument("--kafka-topic", default="lightcurve-points",
                        help="Kafka topic for light-curve points (default: lightcurve-points)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    t_days = np.logspace(np.log10(0.1), np.log10(1000.0), 300)
    rng    = np.random.default_rng(args.seed)

    d_min3, d_max3 = args.d_min ** 3, args.d_max ** 3
    band_labels = list(BANDS.keys())

    fig, ax = plt.subplots(figsize=(11, 6))

    print(f"Generating {args.n_samples} afterglow samples "
          f"(Gaussian jet, iota sine-prior 0–45°, "
          f"d_L {args.d_min}–{args.d_max} Mpc volume-weighted) ...")
    print(f"  Photometry CSVs → {args.output_dir}/")
    if not args.no_kafka:
        print(f"  Kafka streaming → {args.kafka_topic} (bin_size={args.m_bins})")
    print(f"{'#':>3}  {'iota°':>6}  {'d_L':>7}  {'logE0':>6}  {'logn0':>6}  "
          f"{'thetaC':>7}  r_peak")
    print("-" * 60)

    # Initialize Kafka producer (enabled by default, disable with --no-kafka)
    kafka_producer = None
    if args.m_bins > 0:
        try:
            kafka_producer = LightCurveProducer(
                bootstrap_servers=args.kafka_bootstrap_servers,
                topic=args.kafka_topic,
            )
            print(f"Connected to Kafka at {args.kafka_bootstrap_servers}")
        except Exception as e:
            print(f"Warning: Failed to connect to Kafka: {e}")
            print("Continuing without Kafka streaming")
            kafka_producer = None

    start_time = time.time()
    plotted = 0
    for i in range(args.n_samples):
        p = sample_prior(rng)

        # volume-weighted distance
        d_L_mpc = float((rng.uniform() * (d_max3 - d_min3) + d_min3) ** (1.0 / 3.0))
        d_L_cm  = d_L_mpc * MPC_TO_CM

        mags = afterglow_mags(p, t_days, d_L_cm)

        r_label  = band_labels[1]
        r_finite = mags[r_label][np.isfinite(mags[r_label])]
        r_peak   = r_finite.min() if len(r_finite) else np.nan

        iota_deg = np.degrees(p["thetaObs"])
        print(f"{i+1:>3}  {iota_deg:>6.1f}  {d_L_mpc:>7.1f}  "
              f"{p['log10_E0']:>6.1f}  {p['log10_n0']:>6.1f}  "
              f"{np.degrees(p['thetaCore']):>6.1f}°  r={r_peak:.1f}")

        source_id = f"sample_{i+1:04d}"
        
        # Emit points to Kafka as they are generated (real-time binning)
        if kafka_producer:
            for seq_idx, (t, mg, mr) in enumerate(zip(t_days, mags[band_labels[0]], mags[band_labels[1]])):
                # Emit g-band point
                if np.isfinite(mg):
                    kafka_producer.emit_point(
                        source_id=source_id,
                        time=float(t),
                        magnitude=float(mg),
                        band=band_labels[0],
                        sequence=seq_idx,
                        total_points=len(t_days),
                    )
                # Emit r-band point
                if np.isfinite(mr):
                    kafka_producer.emit_point(
                        source_id=source_id,
                        time=float(t),
                        magnitude=float(mr),
                        band=band_labels[1],
                        sequence=seq_idx,
                        total_points=len(t_days),
                    )

        # Dump per-sample CSV: time | g | r
        csv_path = os.path.join(args.output_dir, f"sample_{i+1:04d}.csv")
        mag_g = mags[band_labels[0]]
        mag_r = mags[band_labels[1]]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "# " + "  ".join([
                    f"thetaObs={p['thetaObs']:.4f}",
                    f"log10_E0={p['log10_E0']:.4g}",
                    f"thetaCore={p['thetaCore']:.4f}",
                    f"thetaWing={p['thetaWing']:.4f}",
                    f"log10_n0={p['log10_n0']:.4g}",
                    f"p={p['p']:.4g}",
                    f"log10_eps_e={p['log10_eps_e']:.4g}",
                    f"log10_eps_B={p['log10_eps_B']:.4g}",
                    f"d_L_mpc={d_L_mpc:.4g}",
                ])
            ])
            writer.writerow(["time_days", "g_ps1_mag", "r_ps1_mag"])
            for t, mg, mr in zip(t_days, mag_g, mag_r):
                writer.writerow([f"{t:.6f}", f"{mg:.6f}", f"{mr:.6f}"])

        any_detectable = any(
            np.nanmin(m) < MAG_LIMIT for m in mags.values()
            if np.any(np.isfinite(m))
        )
        if not any_detectable:
            continue

        alpha      = 0.85 if args.n_samples == 1 else 0.35
        lw         = 2.2  if args.n_samples == 1 else 1.0
        dist_alpha = alpha * (1.0 - 0.5 * (d_L_mpc - args.d_min) / (args.d_max - args.d_min))

        for label in band_labels:
            mag  = mags[label]
            good = np.isfinite(mag) & (mag < MAG_LIMIT + 2)
            if good.sum() < 2:
                continue
            ax.plot(t_days[good], mag[good],
                    color=BAND_COLORS[label], lw=lw, alpha=dist_alpha)

        plotted += 1

    elapsed_time = time.time() - start_time

    # Flush Kafka producer
    if kafka_producer:
        print("Flushing Kafka messages...")
        kafka_producer.flush()
        kafka_producer.close()

    print(f"\nPlotted {plotted}/{args.n_samples} detectable samples "
          f"(peak < {MAG_LIMIT} mag).")
    print(f"Generation time: {elapsed_time:.2f}s")

    # ── Legend: one entry per band ────────────────────────────────────────────
    for label in band_labels:
        ax.plot([], [], color=BAND_COLORS[label], lw=2, label=label)
    ax.legend(fontsize=10, loc="lower right")

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Observer time (days post-trigger)", fontsize=12)
    ax.set_ylabel("Apparent magnitude AB (bright at top)", fontsize=12)
    ax.set_title(
        f"GRB afterglow samples — afterglowpy Gaussian jet\n"
        f"n={args.n_samples} prior draws  |  iota sine-prior 0–45°  |  "
        f"d_L {args.d_min}–{args.d_max} Mpc (volume-weighted)",
        fontsize=11,
    )
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")

    # Plotting binned curves

    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Observer time (days post-trigger)", fontsize=12)
    ax.set_ylabel("Apparent magnitude AB (bright at top)", fontsize=12)
    ax.set_title(
        f"GRB afterglow samples — afterglowpy Gaussian jet\n"
        f"n={args.n_samples} prior draws  |  iota sine-prior 0–45°  |  "
        f"d_L {args.d_min}–{args.d_max} Mpc (volume-weighted)",
        fontsize=11,
    )
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
