from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

TEMPERATURE_VARIABLES = {"t2m", "d2m"}

PRESSURE_PRECIPITATION_VARIABLES = {"sp", "tp"}

WIND_VARIABLES = {"u10", "v10"}


def _extract_archive(archive_path: Path, destination: Path) -> list[Path]:
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Expected a ZIP archive from CDS: {archive_path}")

    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.namelist():
            target = (destination / member).resolve()
            if not str(target).startswith(str(destination.resolve())):
                raise ValueError(f"Unsafe archive member: {member}")
        archive.extractall(destination)

    files = sorted(destination.rglob("*.nc"))

    if not files:
        raise ValueError(f"No NetCDF files found in archive: {archive_path}")

    return files


def _read_netcdf(path: Path) -> pd.DataFrame:
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        data = dataset.load()

    frame = data.to_dataframe(dim_order=["valid_time"]).reset_index()

    return frame


def _merge_netcdf_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No NetCDF data frames to merge")

    merged = frames[0].copy()
    merge_columns = ["valid_time", "latitude", "longitude"]

    for frame in frames[1:]:
        merged = merged.merge(frame, on=merge_columns, how="outer")

    required = {"valid_time", "latitude", "longitude", "t2m", "d2m", "sp", "tp", "u10", "v10"}
    missing = required.difference(merged.columns)

    if missing:
        raise ValueError("Missing ERA5 variables: " + ", ".join(sorted(missing)))

    if merged[merge_columns].duplicated().any():
        raise ValueError("Duplicate timestamps after NetCDF merge")

    return merged


def _relative_humidity(temperature_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    a = 17.625
    b = 243.04

    saturation = np.exp(a * temperature_c / (b + temperature_c))
    vapor = np.exp(a * dewpoint_c / (b + dewpoint_c))
    humidity = 100.0 * vapor / saturation

    return humidity.clip(lower=0.0, upper=100.0)


def parse_timeseries(archive_path: str | Path) -> pd.DataFrame:
    archive_path = Path(archive_path)

    if not archive_path.exists():
        raise FileNotFoundError(archive_path)

    with tempfile.TemporaryDirectory(prefix="era5_timeseries_") as temp_dir:
        extracted_files = _extract_archive(archive_path, Path(temp_dir))
        frames = [_read_netcdf(path) for path in extracted_files]
        hourly = _merge_netcdf_frames(frames)

    hourly["valid_time"] = pd.to_datetime(hourly["valid_time"], utc=True)

    hourly["temperature_c"] = hourly["t2m"] - 273.15
    hourly["dewpoint_c"] = hourly["d2m"] - 273.15
    hourly["humidity"] = _relative_humidity(hourly["temperature_c"], hourly["dewpoint_c"])
    hourly["pressure_pa"] = hourly["sp"]
    hourly["precipitation_m"] = hourly["tp"]
    hourly["wind_speed"] = np.sqrt(hourly["u10"] ** 2 + hourly["v10"] ** 2)

    result = hourly[
        [
            "valid_time",
            "latitude",
            "longitude",
            "temperature_c",
            "dewpoint_c",
            "humidity",
            "pressure_pa",
            "precipitation_m",
            "wind_speed",
        ]
    ].copy()

    result = result.sort_values("valid_time").reset_index(drop=True)

    return result
