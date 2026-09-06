from dataclasses import dataclass

BASE_URL = "https://surveilans.kemkes.go.id/assets/publikasi/laporan-mingguan-situasi-terkini-penyakit-infem-dan-potensial-klb-wabah-m{week}-{year}.pdf"


@dataclass(frozen=True)
class SKDRReport:
    year: int
    week: int
    url: str


def build_pdf_url(year: int, week: int) -> str:
    return BASE_URL.format(week=week, year=year)
