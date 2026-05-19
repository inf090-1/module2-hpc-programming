import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def read_results_csv(path: str):
    points = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            threads = int(row["THREADS"])
            algo = row["TYPE"].strip().upper()
            # benchmark.sh writes TIME as a float string
            time_s = float(row["TIME"])
            points.append((threads, algo, time_s))
    return points


def main():
    parser = argparse.ArgumentParser(description="Generate roofline plot for matmul benchmark.")
    parser.add_argument("--results", default="results.csv", help="Path to results.csv from benchmark.sh")
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension N (NxN)")
    parser.add_argument("--peak-flops-gflops", type=float, default=67.0, help="Peak compute (GFLOP/s)")
    parser.add_argument("--peak-bw-gbs", type=float, default=25.0, help="Peak memory bandwidth (GB/s)")
    parser.add_argument("--out", default="figures/matmul_roofline.png", help="Output PNG path")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    results_path = args.results
    if not os.path.isabs(results_path):
        results_path = os.path.join(here, results_path)

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(here, out_path)

    points = read_results_csv(results_path)

    # Matmul arithmetic intensity approximation using compulsory traffic:
    #   FLOPs ~ 2*N^3
    #   Bytes ~ (A + B + C) = 3*N^2 * sizeof(double)
    # sizeof(double) = 8 bytes
    flops = 2.0 * (args.n ** 3)
    bytes_min = 3.0 * (args.n ** 2) * 8.0
    ai = flops / bytes_min

    # Convert measured runtime to achieved GFLOP/s
    achieved = []
    for threads, algo, time_s in points:
        gflops = (flops / time_s) / 1e9
        achieved.append((threads, algo, gflops))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Roofline: y = min(peak_flops, x * peak_bw)
    # Limit AI range to a sensible window around the benchmark's actual AI.
    ai_vals = np.logspace(-1, 3, 500)
    perf_vals = np.minimum(args.peak_flops_gflops, ai_vals * args.peak_bw_gbs)

    plt.figure(figsize=(9, 6))
    plt.plot(ai_vals, perf_vals, color="black", linewidth=2, label="Hardware Roofline")
    plt.plot(
        ai_vals,
        [args.peak_flops_gflops] * len(ai_vals),
        linestyle="--",
        color="gray",
        label=f"Peak Compute ({args.peak_flops_gflops} GF/s)",
    )
    plt.plot(
        ai_vals,
        ai_vals * args.peak_bw_gbs,
        linestyle="-.",
        color="gray",
        label=f"Peak Mem BW ({args.peak_bw_gbs} GB/s)",
    )

    # Scatter points.
    # - Color encodes the implementation (SERIAL, SIMD, BLAS).
    # - Marker shape encodes the thread count (T=1, T=2, T=4).
    color_by_algo = {
        "SERIAL": "tab:green",
        "SIMD": "tab:blue",
        "BLAS": "tab:orange",
    }
    marker_by_threads = {
        1: "x",
        2: "o",
        4: "s",
    }

    for threads, algo, gflops in achieved:
        color = color_by_algo.get(algo, "tab:red")
        marker = marker_by_threads.get(threads, "D")
        label = f"{algo} (T={threads})"
        plt.scatter([ai], [gflops], color=color, marker=marker, s=55, label=label)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title(f"Matmul Roofline (N={args.n}, AI~{ai:.2f})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc="lower right")
    # Tighten y-axis to show the region around peak compute.
    plt.ylim(0.1, args.peak_flops_gflops * 2.0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    main()
