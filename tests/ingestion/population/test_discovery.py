from dengue_mlops.ingestion.population.discovery import candidate_score, is_population_candidate


def test_current_population_table_is_candidate() -> None:
    item = {
        "id": "Mjc4MSMy",
        "title": ("Jumlah Penduduk menurut Provinsi dan Jenis Kelamin"),
        "id_subject": 519,
    }

    assert is_population_candidate(item)
    assert candidate_score(item) > 0


def test_projection_table_is_rejected() -> None:
    item = {
        "id": "MTg4NiMy",
        "title": ("Jumlah Penduduk Hasil Proyeksi Menurut Provinsi dan Jenis Kelamin"),
        "id_subject": 519,
    }

    assert not is_population_candidate(item)


def test_regional_table_is_rejected() -> None:
    item = {
        "id": "x",
        "title": ("Jumlah Penduduk menurut Kabupaten/Kota dan Kelompok Umur"),
        "id_subject": 519,
    }

    assert not is_population_candidate(item)


def test_subject_must_match() -> None:
    item = {
        "id": "x",
        "title": ("Jumlah Penduduk menurut Provinsi dan Jenis Kelamin"),
        "id_subject": 999,
    }

    assert not is_population_candidate(item)
