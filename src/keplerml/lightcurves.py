"""Light curve retrieval and Box Least Squares transit search.

This module reproduces, on raw Kepler photometry, the measurements that the
KOI table reports as `koi_period`, `koi_duration`, and `koi_depth`. It is the
front half of the pipeline: signal in, tabular features out.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.timeseries import BoxLeastSquares

from . import config

PPM = 1e6


@dataclass
class LightCurveData:
    """A stitched, normalized Kepler light curve."""

    target: str
    time: np.ndarray  # barycentric Kepler Julian date, days
    flux: np.ndarray  # normalized to a median of 1.0
    quarters: tuple[int, ...]

    @property
    def baseline_days(self) -> float:
        return float(self.time.max() - self.time.min())

    @property
    def n_points(self) -> int:
        return int(self.time.size)


@dataclass
class BLSResult:
    """Best-fitting box model and the periodogram it came from."""

    period_grid: np.ndarray
    power: np.ndarray
    period: float  # days
    t0: float  # mid-transit time, same units as the light curve
    duration: float  # days
    depth: float  # fractional
    depth_snr: float
    stats: dict

    @property
    def duration_hours(self) -> float:
        return self.duration * 24.0

    @property
    def depth_ppm(self) -> float:
        return self.depth * PPM


def _import_lightkurve():
    """Import lightkurve on demand, quietly.

    The import is deferred so the tabular half of the project runs without the
    dependency, and warnings are suppressed because lightkurve reports on
    optional submodules this project does not use.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import lightkurve

    return lightkurve


def _cache_path(target: str, quarters: list[int]) -> Path:
    slug = target.lower().replace(" ", "")
    quarter_slug = "q" + "-".join(str(q) for q in quarters)
    return config.LIGHTCURVE_CACHE_DIR / f"{slug}_{quarter_slug}.npz"


def fetch_light_curve(
    target: str,
    quarters: list[int] | None = None,
    refresh: bool = False,
) -> LightCurveData:
    """Download and stitch Kepler long-cadence photometry for one target.

    Results are cached as compressed arrays so the notebook only reaches the
    archive once per target. Outlier clipping is asymmetric: upward spikes are
    instrumental, while downward excursions are the transits being searched
    for and must be preserved.
    """
    quarters = list(quarters or config.BLS_QUARTERS)
    cache_file = _cache_path(target, quarters)

    if cache_file.exists() and not refresh:
        cached = np.load(cache_file)
        return LightCurveData(
            target=target,
            time=cached["time"],
            flux=cached["flux"],
            quarters=tuple(quarters),
        )

    lk = _import_lightkurve()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = lk.search_lightcurve(
            target, mission="Kepler", author="Kepler", cadence="long", quarter=quarters
        )
        if len(search) == 0:
            raise ValueError(f"No Kepler long-cadence data found for {target}")

        collection = search.download_all()
        curve = (
            collection.stitch()
            .remove_nans()
            .remove_outliers(sigma_upper=4, sigma_lower=20)
        )

    time = np.asarray(curve.time.value, dtype=float)
    flux = np.asarray(curve.flux.value, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]

    order = np.argsort(time)
    time, flux = time[order], flux[order]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, time=time, flux=flux)

    return LightCurveData(target=target, time=time, flux=flux, quarters=tuple(quarters))


def flatten_flux(
    curve: LightCurveData,
    window_length: int = config.FLATTEN_WINDOW,
    break_tolerance: int = 5,
) -> np.ndarray:
    """Remove slow stellar and instrumental trends with a Savitzky-Golay filter.

    The window is long compared with a transit, so the filter follows spot
    modulation and thermal drift without absorbing the transit itself.
    """
    lk = _import_lightkurve()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lc = lk.LightCurve(time=curve.time, flux=curve.flux)
        flat = lc.flatten(window_length=window_length, break_tolerance=break_tolerance)

    return np.asarray(flat.flux.value, dtype=float)


def point_to_point_scatter(flux: np.ndarray) -> float:
    """Robust per-cadence noise estimate from successive differences.

    Differencing removes any residual trend, and the median absolute deviation
    keeps the estimate from being inflated by the transits themselves.
    """
    differences = np.diff(flux)
    mad = np.median(np.abs(differences - np.median(differences)))
    return float(1.4826 * mad / np.sqrt(2.0))


def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float = config.BLS_MIN_PERIOD,
    max_period: float = config.BLS_MAX_PERIOD,
    period_step: float = config.BLS_PERIOD_STEP,
    durations: list[float] | None = None,
    refine: bool = True,
) -> tuple[BoxLeastSquares, BLSResult]:
    """Search for a periodic box-shaped dip and refine the strongest peak.

    A coarse scan over the full period range locates the peak, then a narrow
    rescan around it recovers the period to a precision the coarse grid cannot
    resolve. Per-cadence uncertainties are supplied so that the reported depth
    signal-to-noise is on a physical scale.
    """
    durations = list(durations or config.BLS_DURATIONS)
    uncertainty = np.full_like(flux, point_to_point_scatter(flux))
    bls = BoxLeastSquares(time, flux, dy=uncertainty)

    period_grid = np.arange(min_period, max_period, period_step)
    periodogram = bls.power(period_grid, durations)
    peak = int(np.argmax(periodogram.power))
    best_period = float(periodogram.period[peak])

    if refine:
        window = max(5 * period_step, 0.002 * best_period)
        fine_grid = np.arange(
            best_period - window, best_period + window, period_step / 20.0
        )
        fine_grid = fine_grid[fine_grid > 0]
        fine = bls.power(fine_grid, durations)
        fine_peak = int(np.argmax(fine.power))
        best = {
            "period": float(fine.period[fine_peak]),
            "t0": float(fine.transit_time[fine_peak]),
            "duration": float(fine.duration[fine_peak]),
            "depth": float(fine.depth[fine_peak]),
            "depth_snr": float(fine.depth_snr[fine_peak]),
        }
    else:
        best = {
            "period": best_period,
            "t0": float(periodogram.transit_time[peak]),
            "duration": float(periodogram.duration[peak]),
            "depth": float(periodogram.depth[peak]),
            "depth_snr": float(periodogram.depth_snr[peak]),
        }

    stats = bls.compute_stats(best["period"], best["duration"], best["t0"])

    result = BLSResult(
        period_grid=np.asarray(periodogram.period, dtype=float),
        power=np.asarray(periodogram.power, dtype=float),
        stats=stats,
        **best,
    )
    return bls, result


def _significance(pair: tuple[float, float]) -> float:
    """Convert an astropy (value, uncertainty) pair into a signed significance."""
    value, error = float(pair[0]), float(pair[1])
    return value / error if error > 0 else np.nan


def transit_features(result: BLSResult) -> dict[str, float]:
    """Turn a BLS detection into the vetting features used for classification.

    Beyond period, depth, and duration, two diagnostics separate planets from
    eclipsing binaries: a dip half an orbit after the primary event, which is
    the companion being occulted, and alternating depths between odd- and
    even-numbered events, which means the true period is twice the detected
    one. Both are reported as a fraction of the primary depth so they can be
    compared across targets of very different brightness.
    """
    stats = result.stats
    primary_depth = abs(result.depth) if result.depth != 0 else np.nan

    depth_odd, odd_error = (float(v) for v in stats["depth_odd"])
    depth_even, even_error = (float(v) for v in stats["depth_even"])
    combined_error = np.hypot(odd_error, even_error)
    odd_even_difference = abs(depth_odd - depth_even)

    secondary_depth = float(stats["depth_phased"][0])

    return {
        "bls_period": result.period,
        "bls_duration_hours": result.duration_hours,
        "bls_depth_ppm": result.depth_ppm,
        "bls_depth_snr": result.depth_snr,
        "bls_duty_cycle": result.duration / result.period,
        "odd_even_ratio": odd_even_difference / primary_depth,
        "odd_even_sigma": (
            odd_even_difference / combined_error if combined_error > 0 else np.nan
        ),
        "secondary_depth_ppm": secondary_depth * PPM,
        "secondary_ratio": secondary_depth / primary_depth,
        "secondary_sigma": _significance(stats["depth_phased"]),
        "n_transits": int(np.sum(np.asarray(stats["per_transit_count"]) > 0)),
    }


def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    phase_min: float = -0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold onto orbital phase over one period, with the transit at phase zero.

    `phase_min` sets where the window starts. The default of -0.5 is centred on
    the transit but places phase 0.5, where a secondary eclipse would appear,
    at both edges of the range and therefore split across them. Passing -0.25
    keeps the primary and the secondary both inside the window.
    """
    offset = -phase_min * period
    phase = ((time - t0 + offset) % period) / period + phase_min
    order = np.argsort(phase)
    return phase[order], flux[order]


def bin_phase(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int = 200,
    phase_min: float = -0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Average the folded curve in equal-width phase bins for plotting."""
    edges = np.linspace(phase_min, phase_min + 1.0, n_bins + 1)
    index = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)

    totals = np.bincount(index, weights=flux, minlength=n_bins)
    counts = np.bincount(index, minlength=n_bins)

    centres = 0.5 * (edges[:-1] + edges[1:])
    occupied = counts > 0
    means = np.full(n_bins, np.nan)
    means[occupied] = totals[occupied] / counts[occupied]
    return centres[occupied], means[occupied]
