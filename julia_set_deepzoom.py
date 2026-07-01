#!/usr/bin/env python3
"""
Julia Set Visualiser — Deep Zoom Edition
─────────────────────────────────────────────────────────────────────
A fixed, high-performance rewrite that resolves two real bugs found in
the previous "advanced" version, plus a full visual-quality pass.
THIS CODE IS BEEN GENERATED FULLY USING CALUDE A.I AS THE PREVIOUS VERSION (julia_set(modified).py) MADE BY MYSELF (MATH) WITH GEMINI A.I HAD FEW ERRORS OR BUGS WHICH NOW IS BEEN RESOLVED USING CLAUDE
BUG #1 — Zoom "returns to a previous point" / gets stuck repeating
────────────────────────────────────────────────────────────────────
The old cycle-detector rejected a new zoom target if it was close to
a *previous* target AND the window width barely changed:

    abs(new_state.width - prev.width) < 1e-8

That 1e-8 is an ABSOLUTE tolerance. Once you're zoomed in deep enough
that the width itself is smaller than ~1e-8 (roughly 18-19 zoom-ins at
3x), EVERY pair of consecutive widths differs by less than 1e-8 simply
because both numbers are tiny — the check is satisfied trivially, no
matter where you click. Combined with users naturally clicking near
the current center to explore one area deeper, this makes the
"cycle detected" false-positive fire almost every time past that
depth, which looks exactly like "it keeps returning to a previous
zoom". Verified numerically: with factor=3 the absolute-diff check
saturates to True at zoom #19 (width ≈ 3.4e-9), long before any real
precision limit is hit.

FIX: compare the RELATIVE width change (a scale-invariant ratio)
instead of an absolute one, and check the *entire* zoom history
instead of just the last five entries.

BUG #2 — Zoom silently degrades near the float64 precision floor
────────────────────────────────────────────────────────────────────
The old code used a single fixed cutoff (`width < 1e-10`) regardless
of resolution or how far the view had drifted from the origin. That's
not adaptive: what actually matters is whether adjacent pixels are
still distinguishable as different float64 numbers, i.e. whether the
per-pixel step is still larger than the local machine epsilon (ULP) at
that coordinate's magnitude. Ignoring that lets the grid silently
collapse (many pixels rounding to the identical complex value), which
renders as a blank/degenerate/repeated-looking image w/o warning.

FIX: compute a resolution- and location-aware "precision margin"
(pixel spacing ÷ local ULP) and refuse to zoom further, with a clear
explanation, once that margin drops below a safety factor — instead
of silently producing garbage.

VISUAL BUG — flat, washed-out coloring near the fractal boundary
────────────────────────────────────────────────────────────────────
The old normalization computed min/max over the WHOLE escape-time
array, including all the interior (never-escaped) points, which are
all pinned at the constant sentinel `max_iter`. For dense interior
regions this constant dominates the normalization range and crushes
the actually-interesting boundary gradient into a thin sliver of the
color scale — the classic "why does my fractal look flat" problem.

FIX: normalize using only the exterior (escaped) pixels, and paint
the interior with a single solid color. This is the standard
"black lake, colorful shore" look used in most fractal renderers,
and it makes the boundary detail pop.

Other changes: squared-magnitude escape test (skips a sqrt per pixel
per iteration), optional 2x2 supersampling with box-filter downsample
for anti-aliasing, optional cyclic/banded coloring, color themes,
robust headless-safe backend selection, and a non-interactive
`render_julia()` entry point for scripting/testing alongside the
original REPL.

Install:
    pip install --user numpy matplotlib

Run (interactive):
    python julia_set_deepzoom.py

Run (one-shot, no prompts):
    python julia_set_deepzoom.py --quick -0.7 0.27015 --theme ember -o julia.png
"""

from __future__ import annotations

import sys
import math
import argparse
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


EPS = np.finfo(np.float64).eps


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Zoom state & precision handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class ZoomState:
    """Immutable viewport state for one zoom level."""
    center_real: float
    center_imag: float
    width: float
    height: float
    iteration: int = 0

    def scale_factor(self) -> float:
        """Zoom scale relative to the initial [-2, 2] view."""
        return 4.0 / self.width

    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max)."""
        hw, hh = self.width / 2.0, self.height / 2.0
        return (
            self.center_real - hw, self.center_real + hw,
            self.center_imag - hh, self.center_imag + hh,
        )

    def pixel_spacing(self, resolution: int) -> float:
        """Coordinate distance covered by one pixel."""
        return self.width / resolution

    def precision_margin(self, resolution: int) -> float:
        """
        How many float64 ULPs "wide" one pixel still is, at this
        coordinate's magnitude. margin <= 1 means neighboring pixels
        would round to the SAME float64 value — the grid degenerates.
        """
        magnitude = max(abs(self.center_real), abs(self.center_imag), 1.0)
        ulp = np.spacing(magnitude)
        return self.pixel_spacing(resolution) / ulp

    def is_precision_safe(self, resolution: int, safety_factor: float = 8.0) -> bool:
        return self.precision_margin(resolution) > safety_factor

    def zoom_into(self, click_real: float, click_imag: float, factor: float = 3.0) -> "ZoomState":
        return ZoomState(
            center_real=click_real,
            center_imag=click_imag,
            width=self.width / factor,
            height=self.height / factor,
            iteration=self.iteration + 1,
        )


@dataclass
class RenderConfig:
    """Rendering configuration."""
    resolution: int = 1024
    max_iter: int = 512
    dpi: int = 100
    colormap: str = "twilight_shifted"
    interior_color: Tuple[float, float, float] = (0.02, 0.02, 0.05)
    cyclic_coloring: bool = False
    cyclic_period: float = 32.0
    supersampling: bool = True
    threshold: float = 2.0

    def __post_init__(self) -> None:
        # The Hubbard–Douady smooth-coloring formula assumes an escape
        # radius >= 2; smaller values can push log2(log2(|z|)) into an
        # undefined domain right at the escape boundary.
        if self.threshold < 2.0:
            print(f"  ⚠ threshold {self.threshold} < 2.0 is unsafe for smooth "
                  f"coloring; using 2.0 instead.")
            self.threshold = 2.0


# Named color themes: (colormap, interior color, cyclic banding)
THEMES = {
    "ember":  dict(colormap="inferno",         interior_color=(0.03, 0.02, 0.04), cyclic_coloring=False),
    "abyss":  dict(colormap="twilight_shifted", interior_color=(0.00, 0.00, 0.03), cyclic_coloring=False),
    "aurora": dict(colormap="viridis",          interior_color=(0.02, 0.05, 0.04), cyclic_coloring=True),
    "candy":  dict(colormap="plasma",           interior_color=(0.05, 0.00, 0.05), cyclic_coloring=True),
    "ocean":  dict(colormap="mako" if "mako" in plt.colormaps() else "cool",
                    interior_color=(0.01, 0.02, 0.05), cyclic_coloring=False),
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fractal engine (vectorized computation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FractalEngine:
    """Vectorized Julia set escape-time computation with smooth coloring."""

    def __init__(self, c: complex, threshold: float = 2.0):
        self.c = c
        self.threshold = threshold

    def compute(self, zoom_state: ZoomState, config: RenderConfig,
                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Returns:
            escape_time: float64 grid of smooth escape counts
                         (meaningless where alive_mask is True)
            alive_mask:  True where the point never escaped (interior)
            num_bounded: count of interior points
            grid_res:    actual (possibly supersampled) grid resolution
        """
        ss = 2 if config.supersampling else 1
        grid_res = config.resolution * ss

        x_min, x_max, y_min, y_max = zoom_state.bounds()
        x = np.linspace(x_min, x_max, grid_res, dtype=np.float64)
        y = np.linspace(y_min, y_max, grid_res, dtype=np.float64)
        Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]

        escape_time = np.full(Z.shape, float(config.max_iter), dtype=np.float64)
        alive = np.ones(Z.shape, dtype=bool)

        total = grid_res * grid_res
        if verbose:
            print(
                f"  [Zoom {zoom_state.iteration}] {grid_res}×{grid_res} = {total:,} points"
                f"{' (2×2 supersampled)' if ss > 1 else ''}\n"
                f"    c = {self.c.real:.6f} {'+' if self.c.imag >= 0 else ''}{self.c.imag:.6f}i\n"
                f"    bounds: [{x_min:.10g}, {x_max:.10g}] × [{y_min:.10g}, {y_max:.10g}]\n"
                f"    max_iter = {config.max_iter} …",
                flush=True,
            )

        thr2 = self.threshold * self.threshold  # avoid sqrt in the hot loop

        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(config.max_iter):
                if not alive.any():
                    break

                Z[alive] = Z[alive] ** 2 + self.c

                mag2 = Z.real * Z.real + Z.imag * Z.imag
                newly_escaped = alive & (mag2 > thr2)

                if newly_escaped.any():
                    abs_z = np.sqrt(mag2[newly_escaped])
                    abs_z = np.maximum(abs_z, 1.0 + 1e-12)  # guard log domain
                    escape_time[newly_escaped] = (i + 1.0) - np.log2(np.log2(abs_z))
                    alive[newly_escaped] = False

                if verbose and (i + 1) % max(1, config.max_iter // 10) == 0:
                    pct = 100.0 * (1.0 - alive.sum() / total)
                    print(f"    [{i + 1}/{config.max_iter}] {pct:.1f}% escaped", flush=True)

        num_bounded = int(alive.sum())
        if verbose:
            pct_bounded = 100.0 * num_bounded / total
            print(f"  Done: {num_bounded:,} / {total:,} bounded ({pct_bounded:.1f}%)\n", flush=True)

        return escape_time, alive, num_bounded, grid_res


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Zoom manager — navigation, history, precision-aware limits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ZoomManager:
    """Manages zoom state, history, and precision-aware cycle detection."""

    def __init__(self, resolution: int = 1024):
        self.resolution = resolution
        self.initial_state = ZoomState(0.0, 0.0, 4.0, 4.0, 0)
        self.zoom_history: list[ZoomState] = [self.initial_state]
        self.current = self.initial_state

    def zoom_in(self, click_real: float, click_imag: float, factor: float = 3.0) -> bool:
        candidate = self.current.zoom_into(click_real, click_imag, factor)

        # ── Precision floor: stop BEFORE the grid degenerates ──────
        if not candidate.is_precision_safe(self.resolution):
            margin = candidate.precision_margin(self.resolution)
            print(
                f"  ⚠ Zoom limit reached: float64 precision is exhausted here.\n"
                f"    Each pixel would only be {margin:.1f}× the smallest "
                f"representable step at this location.\n"
                f"    Zooming further would render a degenerate / repeated image, "
                f"not real detail.\n"
                f"    → zoom out, pick a nearby target, or increase resolution "
                f"to buy a bit more depth."
            )
            return False

        # ── Cycle detection: RELATIVE tolerance, full history ──────
        # (The old absolute-tolerance check saturated to "always equal"
        # once widths themselves were smaller than the tolerance —
        # that's the bug that made deep zooms look like they were
        # snapping back to a previous view.)
        for prev in self.zoom_history:
            dist = abs(
                complex(candidate.center_real, candidate.center_imag)
                - complex(prev.center_real, prev.center_imag)
            )
            rel_width_diff = abs(candidate.width - prev.width) / prev.width
            if dist < candidate.width * 0.01 and rel_width_diff < 1e-6:
                print("  ⚠ Zoom cycle detected — this region has already been rendered.")
                return False

        self.current = candidate
        self.zoom_history.append(candidate)
        return True

    def zoom_out(self) -> bool:
        if len(self.zoom_history) <= 1:
            print("  Already at root zoom level.")
            return False
        self.zoom_history.pop()
        self.current = self.zoom_history[-1]
        return True

    def reset(self) -> None:
        self.zoom_history = [self.initial_state]
        self.current = self.initial_state

    def remaining_zooms(self, factor: float = 3.0) -> int:
        """Rough estimate of how many more factor-x zooms are safe."""
        margin = self.current.precision_margin(self.resolution)
        if margin <= 1:
            return 0
        return max(0, int(math.log(margin, factor)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Color mapper — interior/exterior separation for real contrast
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ColorMapper:

    @staticmethod
    def get_colormap(name: str):
        try:
            return plt.get_cmap(name)
        except ValueError:
            print(f"  ⚠ Colormap '{name}' not found; using 'twilight_shifted'")
            return plt.get_cmap("twilight_shifted")

    @staticmethod
    def normalize_exterior(escape_time: np.ndarray, alive_mask: np.ndarray) -> np.ndarray:
        """
        Log-normalize smooth escape times to [0, 1] using ONLY the
        exterior (escaped) pixels to set the scale. Including the
        interior's constant max_iter sentinel would crush the boundary
        gradient into a sliver of the color range.
        """
        exterior_vals = escape_time[~alive_mask]
        if exterior_vals.size == 0:
            return np.zeros_like(escape_time)

        lo = exterior_vals.min()
        hi_raw = exterior_vals.max()
        shifted = escape_time - lo + 1.0
        log_scaled = np.log(np.maximum(shifted, 1.0))
        hi = np.log(max(hi_raw - lo + 1.0, 1.0 + 1e-12))
        return np.clip(log_scaled / hi, 0.0, 1.0)

    @staticmethod
    def build_rgb(escape_time: np.ndarray, alive_mask: np.ndarray,
                  config: RenderConfig) -> np.ndarray:
        normalized = ColorMapper.normalize_exterior(escape_time, alive_mask)

        if config.cyclic_coloring:
            # Smooth periodic banding (Ultra-Fractal-style rings)
            # applied only to the exterior; interior stays solid.
            phase = (escape_time % config.cyclic_period) / config.cyclic_period
            normalized = np.where(alive_mask, normalized, phase)

        cmap = ColorMapper.get_colormap(config.colormap)
        rgb = cmap(normalized)[..., :3].copy()
        rgb[alive_mask] = config.interior_color
        return rgb


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Visualizer — matplotlib rendering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class JuliaVisualizer:
    """High-quality matplotlib rendering with anti-aliasing and a
    colorbar that reflects the true (exterior-only) gradient scale."""

    def __init__(self, config: RenderConfig):
        self.config = config

    @staticmethod
    def _box_downsample(rgb: np.ndarray, factor: int) -> np.ndarray:
        h, w, ch = rgb.shape
        return rgb.reshape(h // factor, factor, w // factor, factor, ch).mean(axis=(1, 3))

    def render(self, escape_time: np.ndarray, alive_mask: np.ndarray,
               zoom_state: ZoomState, c: complex,
               output_file: Optional[str] = None, show: bool = True) -> None:
        rgb = ColorMapper.build_rgb(escape_time, alive_mask, self.config)

        ss = 2 if self.config.supersampling else 1
        if ss > 1:
            rgb = self._box_downsample(rgb, ss)
        rgb = np.clip(rgb, 0.0, 1.0)

        figsize = (self.config.resolution / self.config.dpi,) * 2
        fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)

        x_min, x_max, y_min, y_max = zoom_state.bounds()
        ax.imshow(rgb, extent=[x_min, x_max, y_min, y_max],
                   origin="lower", interpolation="bicubic")
        ax.set_aspect("equal")

        sign = "+" if c.imag >= 0 else "−"
        ax.set_title(
            f"Julia Set:  c = {c.real:.6f} {sign}{abs(c.imag):.6f}i   "
            f"(zoom {zoom_state.iteration}, ×{zoom_state.scale_factor():.2e})",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Re(z)", fontsize=10)
        ax.set_ylabel("Im(z)", fontsize=10)

        sm = ScalarMappable(cmap=ColorMapper.get_colormap(self.config.colormap),
                             norm=Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04,
                             label="Normalized smooth escape (exterior only)")
        cbar.ax.tick_params(labelsize=8)

        plt.tight_layout()

        if output_file:
            fig.savefig(output_file, dpi=self.config.dpi, bbox_inches="tight", facecolor="white")
            print(f"  ✓ Saved → {output_file}")
        elif show:
            plt.show()

        plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# One-shot convenience function (for scripting / testing)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_julia(c: complex, resolution: int = 1024, max_iter: int = 512,
                  theme: str = "ember", output_file: Optional[str] = None,
                  zoom: Optional[ZoomState] = None, verbose: bool = True) -> None:
    """Render a single Julia set image, no REPL required."""
    config = RenderConfig(resolution=resolution, max_iter=max_iter)
    for k, v in THEMES.get(theme, THEMES["ember"]).items():
        setattr(config, k, v)

    zoom_state = zoom or ZoomState(0.0, 0.0, 4.0, 4.0, 0)
    engine = FractalEngine(c, config.threshold)
    escape_time, alive_mask, _, _ = engine.compute(zoom_state, config, verbose=verbose)

    visualizer = JuliaVisualizer(config)
    visualizer.render(escape_time, alive_mask, zoom_state, c, output_file=output_file)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REPL session
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class JuliaSession:
    PRESETS = {
        "1": (-0.7269, 0.1889, "Douady rabbit"),
        "2": (-0.8, 0.156, "Classic dendrite"),
        "3": (0.285, 0.0, "Filled Julia / Fatou dust"),
        "4": (-0.4, 0.6, "Dragon spiral"),
        "5": (0.0, 0.8, "San Marco fractal"),
        "6": (-0.162, 1.04, "Seahorse valley"),
        "7": (-0.7, 0.27015, "Mini Mandelbrot"),
    }

    def __init__(self) -> None:
        self.c = complex(-0.7, 0.27015)
        self.config = RenderConfig()
        self.theme = "ember"
        self._apply_theme(self.theme)
        self.zoom_manager = ZoomManager(self.config.resolution)
        self.engine = FractalEngine(self.c, self.config.threshold)
        self.visualizer = JuliaVisualizer(self.config)

    def _apply_theme(self, name: str) -> None:
        for k, v in THEMES.get(name, THEMES["ember"]).items():
            setattr(self.config, k, v)

    def choose_c(self) -> bool:
        print("\n" + "=" * 60)
        print("  Select Julia Set Parameter (c)")
        print("=" * 60 + "\n")
        print("  Presets:")
        for k, (r, im, desc) in self.PRESETS.items():
            print(f"    {k})  c = {r:+.6f} {im:+.6f}i  ({desc})")
        print("    0)  Enter custom c")
        print()

        choice = input("  Choose [0–7]: ").strip()
        if choice in self.PRESETS:
            r, im, _ = self.PRESETS[choice]
            self.c = complex(r, im)
            self.engine = FractalEngine(self.c, self.config.threshold)
            return True
        if choice == "0":
            try:
                r = float(input("    Real part: "))
                im = float(input("    Imaginary part: "))
                self.c = complex(r, im)
                self.engine = FractalEngine(self.c, self.config.threshold)
                return True
            except ValueError:
                print("    ✗ Invalid input.")
                return False
        print("    ✗ Invalid choice.")
        return False

    def choose_config(self) -> bool:
        print("\n" + "=" * 60)
        print("  Render Configuration")
        print("=" * 60 + "\n")
        try:
            res = input(f"  Resolution (pixels per side) [{self.config.resolution}]: ").strip()
            if res:
                self.config.resolution = int(res)

            itr = input(f"  Max iterations [{self.config.max_iter}]: ").strip()
            if itr:
                self.config.max_iter = int(itr)

            dpi = input(f"  DPI [{self.config.dpi}]: ").strip()
            if dpi:
                self.config.dpi = int(dpi)

            print("\n  Themes:")
            for i, name in enumerate(THEMES.keys(), 1):
                print(f"    {i})  {name}")
            theme_choice = input(f"\n  Theme (name or number) [{self.theme}]: ").strip()
            if theme_choice:
                names = list(THEMES.keys())
                if theme_choice.isdigit() and 1 <= int(theme_choice) <= len(names):
                    self.theme = names[int(theme_choice) - 1]
                elif theme_choice in THEMES:
                    self.theme = theme_choice
                self._apply_theme(self.theme)

            self.zoom_manager.resolution = self.config.resolution
            self.visualizer = JuliaVisualizer(self.config)
            return True
        except ValueError:
            print("    ✗ Invalid input.")
            return False

    def render_and_explore(self) -> None:
        print("\n" + "─" * 60)
        escape_time, alive_mask, _, _ = self.engine.compute(
            self.zoom_manager.current, self.config,
        )
        self.visualizer.render(escape_time, alive_mask, self.zoom_manager.current, self.c)

        remaining = self.zoom_manager.remaining_zooms()
        print("\n" + "─" * 60)
        print(f"  Zoom Controls:  (≈{remaining} more 3× zooms safe at current precision)")
        print("    z)  Zoom into a point (enter Re, Im coordinates)")
        print("    b)  Zoom back (previous level)")
        print("    r)  Reset to full view")
        print("    m)  Return to main menu")
        print()

        while True:
            cmd = input("  Command [z/b/r/m]: ").strip().lower()

            if cmd == "z":
                try:
                    re = float(input("    Zoom target Re(z): "))
                    im = float(input("    Zoom target Im(z): "))
                    factor_raw = input("    Zoom factor (default 3.0): ").strip()
                    factor = float(factor_raw) if factor_raw else 3.0
                    if self.zoom_manager.zoom_in(re, im, factor):
                        print()
                        self.render_and_explore()
                        break
                except ValueError:
                    print("    ✗ Invalid input.")

            elif cmd == "b":
                if self.zoom_manager.zoom_out():
                    print()
                    self.render_and_explore()
                    break

            elif cmd == "r":
                self.zoom_manager.reset()
                print("  Reset to full view.\n")
                self.render_and_explore()
                break

            elif cmd == "m":
                break

            else:
                print("    ✗ Invalid command.")

    def run(self) -> None:
        print("\n╔" + "═" * 58 + "╗")
        print("║" + "  Julia Set Visualiser — Deep Zoom Edition".center(58) + "║")
        print("╚" + "═" * 58 + "╝\n")

        while True:
            if not self.choose_c():
                continue
            if not self.choose_config():
                continue
            self.render_and_explore()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Julia Set Visualiser — Deep Zoom Edition")
    p.add_argument("--quick", nargs=2, type=float, metavar=("RE", "IM"),
                    help="Render one image non-interactively for c = RE + IM*i")
    p.add_argument("-o", "--output", type=str, default=None, help="Output image path")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--max-iter", type=int, default=512)
    p.add_argument("--theme", type=str, default="ember", choices=list(THEMES.keys()))
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    try:
        if args.quick is not None:
            re, im = args.quick
            render_julia(
                complex(re, im),
                resolution=args.resolution,
                max_iter=args.max_iter,
                theme=args.theme,
                output_file=args.output,
            )
        else:
            JuliaSession().run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
