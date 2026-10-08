"""Project-wide paths, constants, and column definitions."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
LIGHTCURVE_CACHE_DIR = DATA_DIR / "lightcurves"
FIGURE_DIR = PROJECT_ROOT / "figures"

KOI_TABLE_PATH = RAW_DATA_DIR / "koi_cumulative.csv"

for _directory in (RAW_DATA_DIR, LIGHTCURVE_CACHE_DIR, FIGURE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

POSITIVE_CLASS = "CONFIRMED"
NEGATIVE_CLASS = "FALSE POSITIVE"
CLASS_NAMES = ("False positive", "Confirmed planet")

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

IDENTIFIER_COLUMNS = ["kepid", "kepoi_name", "kepler_name"]

# Columns that encode the vetting outcome rather than an independent
# measurement. Training on any of these reproduces the label instead of
# learning from the transit signal, so they are excluded from every model.
LEAKAGE_COLUMNS = [
    "koi_disposition",  # the label itself
    "koi_pdisposition",  # the automated pipeline's disposition
    "koi_score",  # Robovetter confidence in the disposition
    "koi_fpflag_nt",  # "not transit-like" vetting flag
    "koi_fpflag_ss",  # "stellar eclipse" vetting flag
    "koi_fpflag_co",  # "centroid offset" vetting flag
    "koi_fpflag_ec",  # "ephemeris match / contamination" vetting flag
    "koi_vet_stat",  # vetting completion status
    "koi_vet_date",
    "koi_comment",  # free text listing the failed vetting tests
    "koi_disp_prov",  # provenance of the disposition
]

# Transit geometry and signal quality measured directly from the light curve.
TRANSIT_COLUMNS = [
    "koi_period",  # orbital period (days)
    "koi_duration",  # transit duration (hours)
    "koi_depth",  # transit depth (ppm)
    "koi_ror",  # planet-to-star radius ratio
    "koi_dor",  # semi-major axis in stellar radii
    "koi_impact",  # impact parameter
    "koi_incl",  # orbital inclination (degrees)
    "koi_model_snr",  # transit signal-to-noise ratio
    "koi_num_transits",  # number of transits observed
    "koi_max_sngle_ev",  # strongest single-event statistic
    "koi_max_mult_ev",  # multiple-event statistic
    "koi_bin_oedp_sig",  # odd-even depth difference significance
]

# Planet properties derived from the transit fit and the host star.
DERIVED_PLANET_COLUMNS = [
    "koi_prad",  # planet radius (Earth radii)
    "koi_teq",  # equilibrium temperature (K)
    "koi_insol",  # insolation flux (Earth flux)
    "koi_sma",  # semi-major axis (AU)
]

# Host star properties from the Kepler Input Catalog.
STELLAR_COLUMNS = [
    "koi_steff",  # effective temperature (K)
    "koi_slogg",  # surface gravity (log10 cm/s2)
    "koi_srad",  # stellar radius (solar radii)
    "koi_smass",  # stellar mass (solar masses)
    "koi_smet",  # metallicity (dex)
    "koi_srho",  # stellar density fitted from the transit (g/cm3)
    "koi_kepmag",  # Kepler-band magnitude
]

# Astrometric diagnostics: where on the detector the signal originates.
# These are independent measurements, but they test contamination rather than
# transit shape, so the notebook holds them out for a separate ablation.
CENTROID_COLUMNS = [
    "koi_dicco_msky",  # difference-image centroid offset (arcsec)
    "koi_dikco_msky",  # offset relative to the KIC position (arcsec)
    "koi_fwm_stat_sig",  # flux-weighted centroid offset significance
]

# Columns requested from the archive. Keeping this explicit means a schema
# change upstream surfaces as an error rather than a silent column of NaNs.
ARCHIVE_COLUMNS = (
    IDENTIFIER_COLUMNS
    + ["koi_disposition", "koi_pdisposition", "koi_score"]
    + ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]
    + TRANSIT_COLUMNS
    + DERIVED_PLANET_COLUMNS
    + STELLAR_COLUMNS
    + CENTROID_COLUMNS
    + ["koi_time0bk", "koi_count", "koi_fittype"]
)

# ---------------------------------------------------------------------------
# Box Least Squares defaults
# ---------------------------------------------------------------------------

BLS_QUARTERS = [2, 3, 4, 5]
BLS_MIN_PERIOD = 0.5  # days
BLS_MAX_PERIOD = 15.0  # days
BLS_PERIOD_STEP = 1e-4  # days
BLS_DURATIONS = [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30]  # days
FLATTEN_WINDOW = 401  # cadences, roughly 8 days at long cadence
