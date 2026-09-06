from pathlib import Path

import httpx

from .models import DownloadResult


def download_pdf(url: str, output_path: Path, year: int, week: int) -> DownloadResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return DownloadResult(
            year=year, week=week, url=url, status="already_exists", path=output_path
        )

    temp_path = output_path.with_suffix(".tmp")

    try:
        with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as response:
            if response.status_code == 404:
                return DownloadResult(year=year, week=week, url=url, status="not_found")

            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower():
                return DownloadResult(
                    year=year,
                    week=week,
                    url=url,
                    status="failed",
                    error=f"Unexpected Content-Type: {content_type}",
                )

            with temp_path.open("wb") as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)

        temp_path.replace(output_path)

        return DownloadResult(year=year, week=week, url=url, status="success", path=output_path)

    except httpx.RequestError as exc:
        return DownloadResult(year=year, week=week, url=url, status="failed", error=str(exc))

    except httpx.HTTPStatusError as exc:
        return DownloadResult(year=year, week=week, url=url, status="failed", error=str(exc))

    finally:
        if temp_path.exists():
            temp_path.unlink()
