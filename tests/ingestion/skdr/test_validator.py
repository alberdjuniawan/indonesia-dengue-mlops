from dengue_mlops.ingestion.skdr.validator import validate_dengue_rows


def test_validate_valid_rows():
    rows = [
        {"province": province, "cases": index}
        for index, province in enumerate(
            sorted(
                {
                    "ACEH",
                    "BALI",
                    "BANGKA BELITUNG",
                    "BANTEN",
                    "BENGKULU",
                    "DI YOGYAKARTA",
                    "GORONTALO",
                    "JAKARTA",
                    "JAMBI",
                    "JAWA BARAT",
                    "JAWA TENGAH",
                    "JAWA TIMUR",
                    "KALIMANTAN BARAT",
                    "KALIMANTAN SELATAN",
                    "KALIMANTAN TENGAH",
                    "KALIMANTAN TIMUR",
                    "KALIMANTAN UTARA",
                    "KEPULAUAN RIAU",
                    "LAMPUNG",
                    "MALUKU",
                    "MALUKU UTARA",
                    "NUSA TENGGARA BARAT",
                    "NUSA TENGGARA TIMUR",
                    "PAPUA",
                    "PAPUA BARAT",
                    "PAPUA BARAT DAYA",
                    "PAPUA PEGUNUNGAN",
                    "PAPUA SELATAN",
                    "PAPUA TENGAH",
                    "RIAU",
                    "SULAWESI BARAT",
                    "SULAWESI SELATAN",
                    "SULAWESI TENGAH",
                    "SULAWESI TENGGARA",
                    "SULAWESI UTARA",
                    "SUMATERA BARAT",
                    "SUMATERA SELATAN",
                    "SUMATERA UTARA",
                }
            )
        )
    ]

    result = validate_dengue_rows(rows)

    assert result["status"] == "VALID"
    assert result["row_count"] == 38
    assert result["missing_provinces"] == []
    assert result["duplicate_provinces"] == []
    assert result["invalid_rows"] == []


def test_validate_partial_rows():
    rows = [{"province": "ACEH", "cases": 10}]

    result = validate_dengue_rows(rows)

    assert result["status"] == "PARTIAL"
    assert result["row_count"] == 1
    assert "ACEH" not in result["missing_provinces"]
