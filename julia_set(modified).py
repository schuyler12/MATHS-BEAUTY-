#!/usr/bin/env python3
"""
Julia Set Visualiser – efficient, vectorised, smooth-coloured.
Runs on Fedora XFCE (and any Linux desktop).

Quick-start
-----------
1. Install runtime dependencies (once):
       sudo dnf install python3-tkinter          # for the GUI window
       pip install --user numpy matplotlib       # or via dnf

2. Run:
       python julia_set.py
"""

import sys
import numpy as np

# ── Backend selection ──────────────────────────────────────────────
# TkAgg is the most reliable choice on Fedora / XFCE.
# Install python3-tkinter if the window doesn't open:
#     sudo dnf install python3-tkinter
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass  # let matplotlib fall back to whatever is available
import matplotlib.pyplot as plt


# ── Core computation ───────────────────────────────────────────────

def julia_set(
    c: complex,
    num_points: int = 800,
    max_iter: int = 256,
    threshold: float = 2.0,
    colormap: str = "inferno",
    output_file: str | None = None,
) -> None:
    """
    Compute and display the Julia set for  f(z) = z² + c.

    Key design choices vs. the naïve approach
    ──────────────────────────────────────────
    • Fully vectorised NumPy: only one Python-level loop (over iterations),
      all per-point work is done in compiled C via numpy ufuncs.
    • imshow() instead of scatter(): renders a dense pixel grid in
      milliseconds; scatter on 640 000 points is 10–100× slower.
    • Smooth escape-time colouring (Hubbard–Douady formula):
          smooth_i = i + 1 − log₂( log₂|z| )
      This removes the visible banding you get from integer escape counts
      and produces a continuous gradient across the image.
    • Alive-mask update: points that have escaped are excluded from
      further squarings, saving work in later iterations.
    """

    # ── 1. Initial complex grid ────────────────────────────────────
    # Broadcasting trick: row vector (real axis) + col vector (imag axis)
    # → (num_points × num_points) complex matrix without meshgrid overhead.
    x = np.linspace(-2.0, 2.0, num_points, dtype=np.float64)
    y = np.linspace(-2.0, 2.0, num_points, dtype=np.float64)
    Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]   # shape (N, N)

    # ── 2. Iterate ─────────────────────────────────────────────────
    # escape_time holds the smooth iteration count at which each point escaped.
    # Bounded points keep the sentinel value max_iter.
    escape_time = np.full(Z.shape, float(max_iter), dtype=np.float64)
    alive = np.ones(Z.shape, dtype=bool)   # True = point hasn't escaped yet

    total = num_points * num_points
    print(
        f"  Computing {num_points}×{num_points} = {total:,} points  "
        f"(c = {c},  max_iter = {max_iter}) …",
        flush=True,
    )

    for i in range(max_iter):
        # Apply the map only to still-bounded points
        Z[alive] = Z[alive] ** 2 + c

        # Identify newly escaped points
        newly_escaped = alive & (np.abs(Z) > threshold)

        if newly_escaped.any():
            # Smooth colouring: i + 1 – log₂(log₂|z|)
            # |z| > threshold = 2  →  log₂|z| > 1  →  log₂(log₂|z|) > 0
            # so the result is a real number slightly below (i + 1).
            abs_z = np.abs(Z[newly_escaped])
            escape_time[newly_escaped] = (i + 1.0) - np.log2(np.log2(abs_z))
            alive[newly_escaped] = False

        if not alive.any():
            break   # every point has escaped; no need to keep iterating

    bounded = int(alive.sum())
    print(
        f"  Done — {bounded:,} / {total:,} points remain bounded "
        f"({100 * bounded / total:.1f} %).",
        flush=True,
    )

    # ── 3. Render ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))

    sign = "+" if c.imag >= 0 else ""
    ax.set_title(f"Julia Set   c = {c.real:.4f} {sign}{c.imag:.4f}i", fontsize=13)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")

    # imshow treats the array as a pixel grid → very fast for dense outputs
    im = ax.imshow(
        escape_time,
        extent=[-2.0, 2.0, -2.0, 2.0],
        origin="lower",          # y-axis goes up (mathmatically correct)
        cmap=colormap,
        interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Smooth escape count")
    plt.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"  Saved → {output_file}")
    else:
        plt.show()

    plt.close(fig)   # release memory before the next render


# ── CLI helpers ────────────────────────────────────────────────────

# A handful of visually rich starting points
PRESETS: dict[str, tuple[float, float]] = {
    "1": (-0.7269,  0.1889),   # Douady rabbit variant
    "2": (-0.8,     0.156 ),   # Classic dendrite-like
    "3": ( 0.285,   0.0   ),   # Filled Julia / Fatou dust
    "4": (-0.4,     0.6   ),   # Dragon / spiral
    "5": ( 0.0,     0.8   ),   # San Marco fractal
}


def _ask_float(label: str, default: float) -> float:
    """Prompt for a float, accepting Enter for the default."""
    while True:
        raw = input(f"  {label} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print("    ✗  Please enter a number.")


def _ask_int(label: str, default: int) -> int:
    while True:
        raw = input(f"  {label} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print("    ✗  Please enter a whole number.")


# ── Entry point ────────────────────────────────────────────────────

def main() -> None:
    print()
    print("╔══════════════════════════════════╗")
    print("║   Julia Set Visualiser           ║")
    print("╚══════════════════════════════════╝")

    while True:
        print()
        print("── Presets ────────────────────────")
        for k, (r, im_) in PRESETS.items():
            print(f"  {k})  c = {r:+.4f}{im_:+.4f}i")
        print("  0)  Enter custom c")
        print()

        choice = input("Choose preset [1–5] or 0 for custom: ").strip()

        if choice in PRESETS:
            real, imag = PRESETS[choice]
        else:
            print()
            real = _ask_float("Re(c)", -0.7)
            imag = _ask_float("Im(c)",  0.27)

        c_val = complex(real, imag)

        print()
        resolution = _ask_int("Resolution (pixels per axis)", 800)
        max_iter   = _ask_int("Max iterations",               256)

        cmap_raw = input(
            "  Colourmap (inferno / magma / hot / plasma / viridis) [inferno]: "
        ).strip()
        cmap = cmap_raw if cmap_raw else "inferno"

        outfile_raw = input(
            "  Save to file? (e.g. julia.png — leave blank to open window): "
        ).strip()
        outfile = outfile_raw if outfile_raw else None

        print()
        julia_set(
            c_val,
            num_points=resolution,
            max_iter=max_iter,
            colormap=cmap,
            output_file=outfile,
        )

        again = input("\nCompute another? [Y/n]: ").strip().lower()
        if again == "n":
            break

    print("\nBye!\n")


if __name__ == "__main__":
    main()
