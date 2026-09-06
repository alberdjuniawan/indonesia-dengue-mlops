from collections import Counter

EXPECTED_PROVINCES = {
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


def validate_dengue_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "status": "IMAGE_ONLY",
            "row_count": 0,
            "missing_provinces": sorted(EXPECTED_PROVINCES),
            "duplicate_provinces": [],
            "invalid_rows": [],
        }

    provinces = [str(row.get("province", "")).strip().upper() for row in rows]

    duplicate_provinces = [
        province for province, count in (Counter(provinces).items()) if count > 1
    ]

    invalid_rows = []

    for row in rows:
        province = str(row.get("province", "")).strip().upper()

        cases = row.get("cases")

        if not province:
            invalid_rows.append(row)
            continue

        if province not in EXPECTED_PROVINCES:
            invalid_rows.append(row)
            continue

        if not isinstance(cases, int) or cases < 0:
            invalid_rows.append(row)

    missing_provinces = sorted(EXPECTED_PROVINCES - set(provinces))

    if duplicate_provinces or invalid_rows:
        status = "INVALID"

    elif missing_provinces:
        status = "PARTIAL"

    else:
        status = "VALID"

    return {
        "status": status,
        "row_count": len(rows),
        "missing_provinces": missing_provinces,
        "duplicate_provinces": sorted(duplicate_provinces),
        "invalid_rows": invalid_rows,
    }
