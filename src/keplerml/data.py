"""Acquisition and cleaning of the NASA Exoplanet Archive KOI table."""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from . import config

TAP_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
DOWNLOAD_TIMEOUT = 300  # seconds


def _build_tap_url(columns: list[str]) -> str:
    """Return the TAP sync URL for a column-scoped query of the KOI table."""
    query = f"select {','.join(columns)} from cumulative"
    params = urllib.parse.urlencode({"query": query, "format": "csv"})
    return f"{TAP_ENDPOINT}?{params}"


def download_koi_table(
    destination: Path | None = None,
    columns: list[str] | None = None,
) -> Path:
    """Download the cumulative KOI table and write it to `destination`.

    The table is served by the NASA Exoplanet Archive TAP service. Only the
    columns the project uses are requested, which keeps the download near 1 MB.
    """
    destination = Path(destination or config.KOI_TABLE_PATH)
    columns = columns or config.ARCHIVE_COLUMNS

    url = _build_tap_url(columns)
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
        payload = response.read()

    text = payload.decode("utf-8")
    if not text.lstrip().lower().startswith(columns[0].lower()):
        raise RuntimeError(f"Unexpected response from the archive:\n{text[:500]}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def load_koi_table(
    path: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load the KOI table, downloading it once and reusing the cached copy."""
    path = Path(path or config.KOI_TABLE_PATH)
    if refresh or not path.exists():
        download_koi_table(path)
    return pd.read_csv(path, low_memory=False)


def build_labeled_frame(
    koi: pd.DataFrame,
    positive: str = config.POSITIVE_CLASS,
    negative: str = config.NEGATIVE_CLASS,
) -> pd.DataFrame:
    """Reduce the KOI table to the two dispositions used for classification.

    CANDIDATE rows are dropped. Their disposition is unresolved, so they carry
    no ground truth and would add label noise to both training and evaluation.
    """
    labeled = koi[koi["koi_disposition"].isin([positive, negative])].copy()
    labeled["label"] = (labeled["koi_disposition"] == positive).astype(int)
    return labeled.reset_index(drop=True)


def summarize_missingness(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return per-column missing counts and percentages, worst first."""
    missing = frame[columns].isna().sum()
    summary = pd.DataFrame(
        {
            "missing": missing,
            "missing_pct": (100 * missing / len(frame)).round(2),
        }
    )
    return summary.sort_values("missing", ascending=False)


def check_for_leakage(feature_columns: list[str]) -> None:
    """Raise if any vetting-derived column has entered the feature list."""
    leaked = sorted(set(feature_columns) & set(config.LEAKAGE_COLUMNS))
    if leaked:
        raise ValueError(f"Vetting-derived columns present in features: {leaked}")
