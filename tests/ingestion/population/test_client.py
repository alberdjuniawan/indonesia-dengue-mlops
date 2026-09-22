from unittest.mock import Mock

import httpx

from dengue_mlops.ingestion.population.client import BPSClient


def test_list_table_statistics() -> None:
    client = BPSClient(api_key="test-key")
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "status": "OK",
        "data": [
            {"page": 1, "pages": 1, "per_page": 100, "count": 1, "total": 1},
            [
                {
                    "id": "Mjc4MSMy",
                    "title": "Jumlah Penduduk menurut Provinsi dan Jenis Kelamin",
                    "id_subject": 519,
                }
            ],
        ],
    }

    client.client.get = Mock(return_value=response)

    payload = client.list_table_statistics()

    assert payload["status"] == "OK"
    assert payload["data"][1][0]["id"] == "Mjc4MSMy"

    client.close()


def test_view_table_statistics() -> None:
    client = BPSClient(api_key="test-key")
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"status": "OK", "available_years": ["2025"]}

    client.client.get = Mock(return_value=response)

    payload = client.view_table_statistics("Mjc4MSMy")

    assert payload["available_years"] == ["2025"]

    client.close()
