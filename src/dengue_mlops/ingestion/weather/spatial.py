from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from .models import ProvincePoint

PROVINCE_ALIASES = {
    "KEPULAUAN BANGKA BELITUNG": "BANGKA BELITUNG",
    "DAERAH KHUSUS IBUKOTA JAKARTA": "JAKARTA",
    "DKI JAKARTA": "JAKARTA",
    "DAERAH ISTIMEWA YOGYAKARTA": "DI YOGYAKARTA",
    "YOGYAKARTA": "DI YOGYAKARTA",
}


def normalize_province(name: str) -> str:
    normalized = " ".join(str(name).strip().upper().split())
    return PROVINCE_ALIASES.get(normalized, normalized)


def load_province_boundaries(geojson_path: str | Path) -> gpd.GeoDataFrame:
    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(path)
    boundaries = gpd.read_file(path)
    if boundaries.crs is not None and boundaries.crs.to_epsg() != 4326:
        boundaries = boundaries.to_crs(epsg=4326)
    name_candidates = ["PROVINSI", "PROVINCE", "province", "NAME_1", "shapeName", "name", "Name"]
    name_column = next((column for column in name_candidates if column in boundaries.columns), None)
    if name_column is None:
        raise ValueError(
            f"Could not find province-name column. Available columns: {list(boundaries.columns)}"
        )
    boundaries = boundaries.copy()
    boundaries["province"] = boundaries[name_column].astype(str).map(normalize_province)
    boundaries = boundaries[["province", "geometry"]].copy()
    boundaries = boundaries[~boundaries["province"].isna() & (boundaries["province"] != "")]
    boundaries = boundaries.dissolve(by="province", as_index=False)
    boundaries = boundaries.sort_values("province").reset_index(drop=True)
    return boundaries


def build_province_points(boundaries: gpd.GeoDataFrame) -> list[ProvincePoint]:
    if boundaries.empty:
        return []
    points: list[ProvincePoint] = []
    for row in boundaries.itertuples(index=False):
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        representative = geometry.representative_point()
        points.append(
            ProvincePoint(
                province=row.province,
                latitude=float(representative.y),
                longitude=float(representative.x),
            )
        )
    points.sort(key=lambda p: p.province)
    return points
