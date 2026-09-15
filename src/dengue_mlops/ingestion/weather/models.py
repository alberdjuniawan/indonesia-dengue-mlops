from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ProvincePoint:
    province: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class WeatherDaily:
    province: str
    date: date
    temperature: float | None
    rainfall: float | None
    humidity: float | None
    pressure: float | None
    wind_speed: float | None


@dataclass(frozen=True)
class WeatherWeekly:
    province: str
    year: int
    week: int
    temperature: float | None
    rainfall: float | None
    humidity: float | None
    pressure: float | None
    wind_speed: float | None
