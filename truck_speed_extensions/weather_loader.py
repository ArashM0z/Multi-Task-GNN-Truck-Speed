"""ERA5-Land weather feature loader.

The paper conditions on ERA5-Land hourly reanalysis variables aligned to
each road segment's centroid:
  - 2 m temperature
  - 2 m dew point
  - 10 m u/v wind components
  - total precipitation
  - snow depth

This module pulls the relevant variables for a (segment, hour) request and
returns a per-segment feature tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class SegmentMetadata:
    segment_id: str
    centroid_lat: float
    centroid_lon: float


REQUIRED_VARS = (
    "t2m", "d2m", "u10", "v10", "tp", "sd",
)


def load_era5_window(
    era5_path: Path, start: str, end: str, bbox: tuple[float, float, float, float],
) -> xr.Dataset:
    """Load ERA5-Land for a time + spatial window."""
    ds = xr.open_dataset(era5_path)
    ds = ds.sel(time=slice(start, end))
    minlon, minlat, maxlon, maxlat = bbox
    ds = ds.sel(longitude=slice(minlon, maxlon), latitude=slice(maxlat, minlat))
    return ds[list(REQUIRED_VARS)]


def features_for_segments(
    ds: xr.Dataset, segments: list[SegmentMetadata], timestamp: np.datetime64,
) -> np.ndarray:
    """Return (N, 6) per-segment ERA5 features at a given timestamp."""
    snapshot = ds.sel(time=timestamp, method="nearest")
    out = np.zeros((len(segments), len(REQUIRED_VARS)), dtype=np.float32)
    for i, seg in enumerate(segments):
        cell = snapshot.sel(latitude=seg.centroid_lat, longitude=seg.centroid_lon, method="nearest")
        for j, var in enumerate(REQUIRED_VARS):
            out[i, j] = float(cell[var].values)
    return out
