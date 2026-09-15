from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from dengue_mlops.ingestion.weather.client import (
    TIMESERIES_DATASET,
    TIMESERIES_VARIABLES,
    ERA5LandClient,
)


def test_download_timeseries():
    fake_result = MagicMock()

    with patch("dengue_mlops.ingestion.weather.client.cdsapi.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.retrieve.return_value = fake_result

        client = ERA5LandClient()

        path = client.download_timeseries(
            start=date(2026, 1, 1),
            end=date(2026, 1, 31),
            latitude=-7.8,
            longitude=112.6,
            output_path="/tmp/weather.zip",
        )

        assert path.name == "weather.zip"

        mock_client.retrieve.assert_called_once()

        dataset, request = mock_client.retrieve.call_args.args

        assert dataset == TIMESERIES_DATASET
        assert request["variable"] == TIMESERIES_VARIABLES

        assert request["location"] == {"longitude": 112.6, "latitude": -7.8}

        assert request["date"] == ["2026-01-01/2026-01-31"]

        assert request["format"] == "netcdf"

        fake_result.download.assert_called_once_with(path)


def test_download_timeseries_invalid_date_range():
    client = ERA5LandClient.__new__(ERA5LandClient)

    with pytest.raises(ValueError, match="end date must be greater than or equal to start date"):
        client.download_timeseries(
            start=date(2026, 1, 31),
            end=date(2026, 1, 1),
            latitude=-7.8,
            longitude=112.6,
            output_path="/tmp/weather.zip",
        )


def test_download_timeseries_invalid_latitude():
    client = ERA5LandClient.__new__(ERA5LandClient)

    with pytest.raises(ValueError, match="latitude must be between -90 and 90"):
        client.download_timeseries(
            start=date(2026, 1, 1),
            end=date(2026, 1, 31),
            latitude=91.0,
            longitude=112.6,
            output_path="/tmp/weather.zip",
        )


def test_download_timeseries_invalid_longitude():
    client = ERA5LandClient.__new__(ERA5LandClient)

    with pytest.raises(ValueError, match="longitude must be between -180 and 180"):
        client.download_timeseries(
            start=date(2026, 1, 1),
            end=date(2026, 1, 31),
            latitude=-7.8,
            longitude=181.0,
            output_path="/tmp/weather.zip",
        )
