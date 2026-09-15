from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://raw.githubusercontent.com/AlfianAliM/Indonesia-GeoJSON/master/provinsi.geojson"
OUTPUT = Path("data/raw/boundaries/indonesia_provinces.geojson")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={"User-Agent": "indonesia-dengue-mlops"})
    print(f"Downloading boundary data from:\n{URL}")
    with urlopen(request, timeout=30) as response:
        content = response.read()
    OUTPUT.write_bytes(content)
    print(f"Saved: {OUTPUT}")
    print(f"Size : {OUTPUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
