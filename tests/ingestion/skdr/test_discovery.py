from dengue_mlops.ingestion.skdr.discovery import build_pdf_url


def test_build_pdf_url():
    url = build_pdf_url(2026, 34)

    assert url.endswith(
        "laporan-mingguan-situasi-terkini-penyakit-infem-dan-potensial-klb-wabah-m34-2026.pdf"
    )
