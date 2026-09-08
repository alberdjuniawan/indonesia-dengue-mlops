from __future__ import annotations

import re

import numpy as np
import pandas as pd

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


def normalize_text(text: str) -> str:
    text = str(text).upper().strip()

    replacements = {
        "SUMA TERA": "SUMATERA",
        "KALIMANTAN TMUR": "KALIMANTAN TIMUR",
        "KALIMANTANUTARA": "KALIMANTAN UTARA",
        "SULAINESI": "SULAWESI",
        "SULAVVESI": "SULAWESI",
        "PAPUA PEGUNUNG-N": "PAPUA PEGUNUNGAN",
        "DKI JAKARTA": "JAKARTA",
        "KEPULAUAN BANGKA BELITUNG": "BANGKA BELITUNG",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^A-Z0-9 ]+", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_trigrams(text: str) -> set[str]:
    text = f"  {text}  "

    return {text[i : i + 3] for i in range(len(text) - 2)}


def province_similarity(a: str, b: str) -> float:
    A = build_trigrams(a)
    B = build_trigrams(b)

    if not A or not B:
        return 0.0

    return len(A & B) / len(A | B)


def match_province(text: str, threshold: float = 0.55, margin: float = 0.05):
    normalized = normalize_text(text)

    if not normalized:
        return None, 0.0

    if normalized == "JAKARTA":
        return "JAKARTA", 1.0

    if normalized == "BANGKA BELITUNG":
        return "BANGKA BELITUNG", 1.0

    if normalized in EXPECTED_PROVINCES:
        return normalized, 1.0

    candidates = []

    for province in EXPECTED_PROVINCES:
        score = province_similarity(normalized, province)

        candidates.append((province, score))

    candidates.sort(key=lambda item: item[1], reverse=True)

    if not candidates:
        return None, 0.0

    best_name, best_score = candidates[0]

    second_score = candidates[1][1] if len(candidates) > 1 else 0.0

    if best_score < threshold:
        return None, best_score

    if len(candidates) > 1 and best_score - second_score < margin:
        return None, best_score

    return best_name, best_score


def parse_easyocr_detection(detection):
    if not isinstance(detection, (list, tuple)):
        return None

    if len(detection) < 3:
        return None

    try:
        bbox = np.asarray(detection[0], dtype=np.float32)

        text = str(detection[1])
        confidence = float(detection[2])

    except (TypeError, ValueError):
        return None

    if bbox.shape != (4, 2):
        return None

    return (bbox, text, confidence)


def merge_number_candidates(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()

    return raw_df.sort_values(["center_y", "center_x"]).reset_index(drop=True)


def build_row_centers(number_df: pd.DataFrame):
    if number_df.empty:
        return []

    centers = []

    for _, row in number_df.sort_values("center_y").iterrows():
        y = float(row["center_y"])

        best_index = None
        best_distance = float("inf")

        for index, existing_y in enumerate(centers):
            distance = abs(y - existing_y)

            if distance <= 12 and distance < best_distance:
                best_index = index
                best_distance = distance

        if best_index is None:
            centers.append(y)
        else:
            centers[best_index] = (centers[best_index] + y) / 2

    return sorted(centers)


def build_rows_from_candidates(province_df: pd.DataFrame, number_df: pd.DataFrame) -> list[dict]:
    if province_df.empty:
        return []

    if number_df.empty:
        return [
            {
                "province": row["province"],
                "cases": None,
                "province_ocr": row["ocr_text"],
                "province_confidence": float(row["confidence"]),
                "province_score": float(row["province_score"]),
                "number_confidence": None,
                "number_preprocess": None,
                "y_distance": None,
                "status": "MISSING_NUMBER",
            }
            for _, row in province_df.iterrows()
        ]

    provinces = province_df.copy().sort_values("center_y").reset_index(drop=True)

    numbers = number_df.copy().sort_values(["center_y", "center_x"]).reset_index(drop=True)

    if "x_right" not in provinces.columns:
        if "x2" in provinces.columns:
            provinces["x_right"] = provinces["x2"]
        else:
            provinces["x_right"] = provinces["center_x"]

    used_numbers = set()
    results = []

    for _, province in provinces.iterrows():
        province_name = str(province["province"])

        province_y = float(province["center_y"])

        province_right = float(province["x_right"])

        candidates = []

        for number_index, number in numbers.iterrows():
            if number_index in used_numbers:
                continue

            number_x = float(number["center_x"])

            number_y = float(number["center_y"])

            if number_x < province_right - 10:
                continue

            y_distance = abs(number_y - province_y)

            candidates.append({"index": number_index, "number": number, "y_distance": y_distance})

        if not candidates:
            results.append(
                {
                    "province": province_name,
                    "cases": None,
                    "province_ocr": province["ocr_text"],
                    "province_confidence": float(province["confidence"]),
                    "province_score": float(province["province_score"]),
                    "number_confidence": None,
                    "number_preprocess": None,
                    "y_distance": None,
                    "status": "MISSING_NUMBER",
                }
            )

            continue

        candidates.sort(key=lambda item: (item["y_distance"], -float(item["number"]["confidence"])))

        best = candidates[0]

        best_distance = float(best["y_distance"])

        number_index = best["index"]
        number = best["number"]

        if best_distance > 25:
            results.append(
                {
                    "province": province_name,
                    "cases": None,
                    "province_ocr": province["ocr_text"],
                    "province_confidence": float(province["confidence"]),
                    "province_score": float(province["province_score"]),
                    "number_confidence": None,
                    "number_preprocess": None,
                    "y_distance": best_distance,
                    "status": "MISSING_NUMBER",
                }
            )

            continue

        used_numbers.add(number_index)

        results.append(
            {
                "province": province_name,
                "cases": int(number["value"]),
                "province_ocr": province["ocr_text"],
                "province_confidence": float(province["confidence"]),
                "province_score": float(province["province_score"]),
                "number_confidence": float(number["confidence"]),
                "number_preprocess": number.get("preprocess"),
                "y_distance": best_distance,
                "status": "OCR",
            }
        )

    return results
