"""Detecting exoplanets in Kepler transit data.

The package is organised around the two halves of the pipeline:

- `data`, `features`, and `modeling` handle the tabular KOI catalogue.
- `lightcurves` recovers the same transit measurements from raw photometry
  using a Box Least Squares periodogram.

`config` holds shared paths and column definitions, `viz` holds plotting.
"""

from . import config, data, features, lightcurves, modeling, viz

__all__ = ["config", "data", "features", "lightcurves", "modeling", "viz"]
__version__ = "1.0.0"
