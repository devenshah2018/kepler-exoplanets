"""Feature engineering for the KOI transit table.

Every engineered feature encodes a physical test that astronomers apply by
hand when vetting a transit signal, so the model has a chance to learn the
same reasoning from data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Mean density of the Sun in g/cm3, used to convert catalog mass and radius
# into a stellar density.
SOLAR_DENSITY = 1.408

HOURS_PER_DAY = 24.0

# Heavy-tailed quantities. A KOI period spans four orders of magnitude and a
# depth spans six, so distance- and margin-based models need them on a log
# scale to behave sensibly.
LOG_TRANSFORM_COLUMNS = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_ror",
    "koi_dor",
    "koi_model_snr",
    "koi_num_transits",
    "koi_max_sngle_ev",
    "koi_max_mult_ev",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_sma",
    "koi_srad",
    "koi_smass",
    "koi_srho",
]

ENGINEERED_COLUMNS = [
    "log_duty_cycle",
    "log_duration_ratio",
    "log_density_ratio",
    "log_snr_per_transit",
]


def _safe_log10(values: pd.Series) -> pd.Series:
    """Base-10 log that maps non-positive and non-finite input to NaN.

    A handful of catalog rows carry a zero radius or zero transit count, which
    would otherwise propagate an infinity through the whole feature matrix.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    finite_positive = numeric.where(np.isfinite(numeric) & (numeric > 0))
    return np.log10(finite_positive)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the four physics-motivated features used alongside the raw columns."""
    out = frame.copy()

    duration_days = out["koi_duration"] / HOURS_PER_DAY

    # Fraction of the orbit spent in transit. A planet crossing a Sun-like star
    # occupies well under 10 percent of its orbit; eclipsing binaries and
    # long-period variability sit much higher.
    out["log_duty_cycle"] = _safe_log10(duration_days / out["koi_period"])

    # Duration expected for a central transit at the fitted scaled semi-major
    # axis, in days. Ratios far from one mean the measured duration is
    # inconsistent with the fitted orbit.
    expected_duration = out["koi_period"] / (np.pi * out["koi_dor"])
    out["log_duration_ratio"] = _safe_log10(duration_days / expected_duration)

    # Stellar density implied by the transit fit divided by the density implied
    # by the catalog mass and radius. A self-consistent planet transit gives a
    # ratio near one; a blended or grazing binary does not.
    catalog_density = SOLAR_DENSITY * out["koi_smass"] / out["koi_srad"] ** 3
    out["log_density_ratio"] = _safe_log10(out["koi_srho"] / catalog_density)

    # Detection strength of a single event. Separates a handful of deep events
    # from a long series of shallow ones that co-add to the same total SNR.
    per_transit = out["koi_model_snr"] / np.sqrt(out["koi_num_transits"])
    out["log_snr_per_transit"] = _safe_log10(per_transit)

    return out


def apply_log_transforms(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Replace skewed columns with their base-10 logs.

    Returns the transformed frame and a mapping from original to new name so
    downstream feature lists can be rewritten consistently.
    """
    columns = columns or LOG_TRANSFORM_COLUMNS
    out = frame.copy()
    renames: dict[str, str] = {}

    for column in columns:
        if column not in out.columns:
            continue
        new_name = f"log_{column.removeprefix('koi_')}"
        out[new_name] = _safe_log10(out[column])
        renames[column] = new_name

    return out, renames


def resolve_feature_names(columns: list[str], renames: dict[str, str]) -> list[str]:
    """Map a raw column list through the log-transform renaming."""
    return [renames.get(column, column) for column in columns]


def build_feature_sets(renames: dict[str, str]) -> dict[str, list[str]]:
    """Define the three nested feature sets compared in the notebook."""
    transit = resolve_feature_names(config.TRANSIT_COLUMNS, renames)
    planet = resolve_feature_names(config.DERIVED_PLANET_COLUMNS, renames)
    stellar = resolve_feature_names(config.STELLAR_COLUMNS, renames)
    centroid = resolve_feature_names(config.CENTROID_COLUMNS, renames)

    physical = transit + planet + stellar + ENGINEERED_COLUMNS

    return {
        "transit_only": transit,
        "physical": physical,
        "physical_plus_centroid": physical + centroid,
    }
