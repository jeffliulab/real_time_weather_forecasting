"""
Fetch real-time HRRR analysis data from NOAA AWS S3 via Herbie.

Downloads 42 individual GRIB2 messages (one per input channel),
extracts the New England subgrid (450x449), and stacks them into
the model's expected input format.
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np

from var_mapping import HRRR_MAPPING, NE_Y_SLICE, NE_X_SLICE

logger = logging.getLogger(__name__)


def find_latest_hrrr_cycle(max_lookback_hours: int = 6) -> datetime:
    """
    Find the most recent HRRR cycle that is available on AWS S3.
    HRRR data typically becomes available ~45-90 minutes after valid time.

    Returns a tz-naive datetime in UTC (Herbie requirement).
    """
    from herbie import Herbie

    now = datetime.now(timezone.utc).replace(tzinfo=None)  # tz-naive UTC
    for hours_ago in range(2, max_lookback_hours + 1):
        cycle_time = (now - timedelta(hours=hours_ago)).replace(
            minute=0, second=0, microsecond=0
        )
        try:
            H = Herbie(
                cycle_time,
                model="hrrr",
                product="sfc",
                fxx=0,
                verbose=False,
            )
            # Check if the index file exists (fast check without downloading data)
            if H.idx is not None:
                logger.info(f"Found HRRR cycle: {cycle_time:%Y-%m-%d %H:%M UTC}")
                return cycle_time
        except Exception:
            continue

    raise RuntimeError(
        f"No HRRR data available in the last {max_lookback_hours} hours. "
        "NOAA servers may be temporarily unavailable."
    )


def _fetch_single_variable(cycle_time: datetime, mapping: dict) -> np.ndarray:
    """
    Fetch one variable from HRRR and extract the NE subgrid.

    Returns:
        np.ndarray of shape (450, 449)
    """
    from herbie import Herbie

    H = Herbie(
        cycle_time,
        model="hrrr",
        product=mapping["product"],
        fxx=mapping["fxx"],
        verbose=False,
    )

    ds = H.xarray(mapping["search"], remove_grib=True)

    # xarray dataset may have different variable names depending on the GRIB message.
    # Get the first data variable (excluding coordinates).
    data_vars = [v for v in ds.data_vars if v not in ("latitude", "longitude", "gribfile_projection")]
    if not data_vars:
        raise ValueError(f"No data variable found for {mapping['name']}")

    field = ds[data_vars[0]].values  # Full CONUS grid

    # Extract NE subgrid
    subgrid = field[NE_Y_SLICE, NE_X_SLICE]

    if subgrid.shape != (450, 449):
        raise ValueError(
            f"Unexpected shape {subgrid.shape} for {mapping['name']}, expected (450, 449)"
        )

    return subgrid.astype(np.float32)


def fetch_hrrr_input(
    cycle_time: datetime = None,
    progress_callback=None,
) -> tuple[np.ndarray, datetime]:
    """
    Fetch all 42 HRRR channels and stack into model input format.

    Args:
        cycle_time: Specific HRRR cycle to fetch. If None, finds the latest.
        progress_callback: Optional callable(fraction, description) for progress updates.

    Returns:
        (input_array, cycle_time) where input_array is (450, 449, 42) float32.
    """
    if cycle_time is None:
        if progress_callback:
            progress_callback(0.05, "Finding latest HRRR cycle...")
        cycle_time = find_latest_hrrr_cycle()

    channels = []
    n_channels = len(HRRR_MAPPING)
    failed = []

    for i, mapping in enumerate(HRRR_MAPPING):
        if progress_callback:
            frac = 0.1 + 0.85 * (i / n_channels)
            progress_callback(frac, f"Fetching {mapping['name']} ({i+1}/{n_channels})...")

        try:
            field = _fetch_single_variable(cycle_time, mapping)
            channels.append(field)
        except Exception as e:
            logger.warning(f"Failed to fetch {mapping['name']}: {e}")
            failed.append(mapping["name"])
            # Fill with zeros as fallback for individual missing channels
            channels.append(np.zeros((450, 449), dtype=np.float32))

    if failed:
        logger.warning(f"Failed channels ({len(failed)}/{n_channels}): {failed}")
        if len(failed) > n_channels // 2:
            raise RuntimeError(
                f"Too many channels failed ({len(failed)}/{n_channels}). "
                "HRRR data may be unavailable."
            )

    input_array = np.stack(channels, axis=-1)  # (450, 449, 42)

    if progress_callback:
        progress_callback(1.0, "Data fetch complete!")

    return input_array, cycle_time
