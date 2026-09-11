from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DownloadStatus = Literal["success", "already_exists", "not_found", "failed"]


@dataclass(frozen=True)
class DownloadResult:
    year: int
    week: int
    url: str
    status: DownloadStatus
    path: Path | None = None
    error: str | None = None
