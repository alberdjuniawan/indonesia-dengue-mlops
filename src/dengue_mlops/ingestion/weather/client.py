from __future__ import annotations

from datetime import date
from pathlib import Path

import cdsapi

TIMESERIES_DATASET = "reanalysis-era5-land-timeseries"

TIMESERIES_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]


class ERA5LandClient:
    def __init__(self) -> None:
        self.client = cdsapi.Client()

    @staticmethod
    def _validate_date_range(start: date, end: date) -> None:
        if end < start:
            raise ValueError("end date must be greater than or equal to start date")

    @staticmethod
    def _validate_coordinate(latitude: float, longitude: float) -> None:
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("latitude must be between -90 and 90")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError("longitude must be between -180 and 180")

    def download_timeseries(self, *, start: date, end: date, latitude: float, longitude: float, output_path: str | Path) -> Path:  # fmt: skip
        self._validate_date_range(start, end)
        self._validate_coordinate(latitude, longitude)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request = {
            "variable": TIMESERIES_VARIABLES,
            "location": {"longitude": float(longitude), "latitude": float(latitude)},
            "date": [f"{start.isoformat()}/{end.isoformat()}"],
            "format": "netcdf",
        }
        result = self.client.retrieve(TIMESERIES_DATASET, request)
        result.download(output_path)
        return output_path
