from dengue_mlops.ingestion.population.parser import parse_population


def test_parse_population_uses_total_and_excludes_indonesia() -> None:
    payload = {
        "var": [
            {
                "val": 2781,
                "label": "Jumlah Penduduk menurut Provinsi dan Jenis Kelamin",
                "unit": "",
                "subj": "Kependudukan",
                "def": "",
                "decimal": 0,
            }
        ],
        "turvar": [
            {"val": 583, "label": "Laki-Laki"},
            {"val": 584, "label": "Perempuan"},
            {"val": 585, "label": "Jumlah "},
        ],
        "tahun": [{"val": 125, "label": "2025"}],
        "turtahun": [{"val": 0, "label": "Tahun"}],
        "vervar": [
            {"val": 1100, "label": "ACEH"},
            {"val": 3100, "label": "DKI JAKARTA"},
            {"val": 9999, "label": "INDONESIA"},
        ],
        "datacontent": {
            "110027815831250": 2828424,
            "110027815841250": 2803336,
            "110027815851250": 5631760,
            "310027815831250": 5371393,
            "310027815841250": 5352941,
            "310027815851250": 10724334,
            "999927815851250": 284667253,
        },
    }

    rows = parse_population(payload)

    assert len(rows) == 2

    assert rows[0].province == "ACEH"
    assert rows[0].year == 2025
    assert rows[0].population == 5_631_760

    assert rows[1].province == "DKI JAKARTA"
    assert rows[1].year == 2025
    assert rows[1].population == 10_724_334
