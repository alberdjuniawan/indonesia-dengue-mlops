from __future__ import annotations

import os
import re
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd
import pymupdf
import torch
from paddleocr import LayoutDetection

from .ocr_parser import build_rows_from_candidates, match_province, parse_easyocr_detection
from .page_locator import find_dengue_pages
from .parser import parse_dengue_page

RENDER_SCALE = 2.0
CHART_PADDING = 25

DEFAULT_TARGET_CHART_INDEX = 1
CHART_MIN_SCORE = 0.70
MASK_OVERLAP_THRESHOLD = 0.50

OCR_MIN_CONFIDENCE = 0.15


NUMBER_RESIZE_SCALE = 3
NUMBER_THRESHOLD = 40

ENABLE_COMPONENT_FILTER = True

DILATION_KERNEL_SIZE = (2, 2)
DILATION_ITERATIONS = 1

ROW_SPLIT_Y_GAP = 5

MIN_COMPONENT_HEIGHT = 2
MIN_COMPONENT_AREA = 3

NUMBER_HORIZONTAL_GAP = 35


ENABLE_PROVINCE_ROW_RECOVERY = True

PROVINCE_RECOVERY_MAX_Y_DISTANCE = 12
PROVINCE_RECOVERY_MIN_SCORE = 0.55

PROVINCE_RECOVERY_HALF_HEIGHT_MIN = 6
PROVINCE_RECOVERY_HALF_HEIGHT_MAX = 12

PROVINCE_RECOVERY_X_PADDING = 10


TEXT_PARSER_MIN_PROVINCES = 35
TEXT_PARSER_REQUIRE_COMPLETE = True

EXPECTED_PROVINCE_COUNT = 38


GPU_ENABLED = os.getenv("DENGUE_OCR_GPU", "1") == "1" and torch.cuda.is_available()


@lru_cache(maxsize=1)
def get_layout_model():
    return LayoutDetection(enable_mkldnn=False)


@lru_cache(maxsize=1)
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=GPU_ENABLED)


def intersection_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)

    bx1, by1, bx2, by2 = map(float, b)

    x1 = max(ax1, bx1)

    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)

    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def overlap_ratio(region, chart) -> float:
    rx1, ry1, rx2, ry2 = map(float, region)

    region_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)

    if region_area <= 0:
        return 0.0

    return intersection_area(region, chart) / region_area


def render_page(pdf_path: Path, page_number: int) -> np.ndarray:
    with pymupdf.open(pdf_path) as document:
        if page_number < 1 or page_number > len(document):
            raise ValueError(
                f"Invalid page {page_number} for {pdf_path}. Total pages: {len(document)}"
            )

        page = document[page_number - 1]

        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE), alpha=False)

        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)


def detect_layout(image: np.ndarray) -> pd.DataFrame:
    model = get_layout_model()

    output = model.predict(image)

    columns = ["id", "label", "score", "x1", "y1", "x2", "y2"]

    if not output:
        return pd.DataFrame(columns=columns)

    boxes = output[0]["boxes"]

    rows = []

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box["coordinate"]]

        rows.append(
            {
                "id": index,
                "label": str(box["label"]),
                "score": float(box["score"]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def select_target_chart(
    layout_df: pd.DataFrame, target_chart_index: int = (DEFAULT_TARGET_CHART_INDEX)
) -> pd.Series:
    charts = layout_df[
        (layout_df["label"].astype(str).str.lower() == "chart")
        & (layout_df["score"] >= CHART_MIN_SCORE)
    ].copy()

    if charts.empty:
        raise RuntimeError("No chart region detected.")

    charts["area"] = (charts["x2"] - charts["x1"]) * (charts["y2"] - charts["y1"])

    charts = charts.sort_values("area", ascending=False).reset_index(drop=True)

    if 0 <= target_chart_index < len(charts):
        return charts.iloc[target_chart_index]

    return charts.iloc[0]


def crop_target_chart(image, chart):
    x1 = round(chart["x1"])

    y1 = round(chart["y1"])

    x2 = round(chart["x2"])

    y2 = round(chart["y2"])

    chart_bbox = (x1, y1, x2, y2)

    px1 = max(0, x1 - CHART_PADDING)

    py1 = max(0, y1 - CHART_PADDING)

    px2 = min(image.shape[1], x2 + CHART_PADDING)

    py2 = min(image.shape[0], y2 + CHART_PADDING)

    crop_bbox = (px1, py1, px2, py2)

    crop = image[py1:py2, px1:px2].copy()

    return (crop, chart_bbox, crop_bbox)


def mask_non_chart_regions(chart_crop, layout_df, chart_bbox, crop_bbox):
    cleaned = chart_crop.copy()

    px1, py1, _, _ = crop_bbox

    masked_regions = []

    for _, region in layout_df.iterrows():
        label = str(region["label"])

        if label.lower() == "chart":
            continue

        region_bbox = (region["x1"], region["y1"], region["x2"], region["y2"])

        overlap = overlap_ratio(region_bbox, chart_bbox)

        if overlap < MASK_OVERLAP_THRESHOLD:
            continue

        lx1 = round(region["x1"] - px1)

        ly1 = round(region["y1"] - py1)

        lx2 = round(region["x2"] - px1)

        ly2 = round(region["y2"] - py1)

        lx1 = max(0, min(lx1, cleaned.shape[1]))

        ly1 = max(0, min(ly1, cleaned.shape[0]))

        lx2 = max(0, min(lx2, cleaned.shape[1]))

        ly2 = max(0, min(ly2, cleaned.shape[0]))

        if lx2 <= lx1 or ly2 <= ly1:
            continue

        cleaned[ly1:ly2, lx1:lx2] = 255

        masked_regions.append(
            {
                "id": int(region["id"]),
                "label": label,
                "score": float(region["score"]),
                "overlap": float(overlap),
                "bbox": (lx1, ly1, lx2, ly2),
            }
        )

    return (cleaned, masked_regions)


def extract_province_candidates(cleaned_chart) -> pd.DataFrame:
    reader = get_ocr_reader()

    gray = cv2.cvtColor(cleaned_chart, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    _, light_threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    images = {"original": cleaned_chart, "gray": gray, "light_threshold": light_threshold}

    rows = []

    for preprocess_name, image in images.items():
        detections = reader.readtext(image, detail=1, paragraph=False)

        scale = 1 if preprocess_name == "original" else 2

        for detection in detections:
            parsed = parse_easyocr_detection(detection)

            if parsed is None:
                continue

            bbox, text, confidence = parsed

            if confidence < OCR_MIN_CONFIDENCE:
                continue

            province, score = match_province(text)

            if province is None:
                continue

            x1 = float(bbox[:, 0].min()) / scale

            y1 = float(bbox[:, 1].min()) / scale

            x2 = float(bbox[:, 0].max()) / scale

            y2 = float(bbox[:, 1].max()) / scale

            rows.append(
                {
                    "province": province,
                    "ocr_text": str(text),
                    "confidence": float(confidence),
                    "province_score": float(score),
                    "center_x": (x1 + x2) / 2,
                    "center_y": (y1 + y2) / 2,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "x_left": x1,
                    "x_right": x2,
                    "y_top": y1,
                    "y_bottom": y2,
                    "preprocess": preprocess_name,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "province",
                "ocr_text",
                "confidence",
                "province_score",
                "center_x",
                "center_y",
                "x1",
                "y1",
                "x2",
                "y2",
                "x_left",
                "x_right",
                "y_top",
                "y_bottom",
                "preprocess",
            ]
        )

    result = pd.DataFrame(rows)

    return (
        result.sort_values(
            ["province", "province_score", "confidence"], ascending=[True, False, False]
        )
        .drop_duplicates(subset=["province"], keep="first")
        .sort_values("center_y")
        .reset_index(drop=True)
    )


def recover_missing_province_candidates(
    cleaned_chart: np.ndarray, province_df: pd.DataFrame, number_df: pd.DataFrame
) -> pd.DataFrame:
    if not ENABLE_PROVINCE_ROW_RECOVERY:
        return province_df.copy()

    if province_df.empty or number_df.empty:
        return province_df.copy()

    current = province_df.copy()

    existing_provinces = set(current["province"].astype(str))

    max_recoveries = max(0, EXPECTED_PROVINCE_COUNT - len(existing_provinces))

    if max_recoveries == 0:
        return current

    province_centers = [float(value) for value in current["center_y"].tolist()]

    if "x2" in current.columns:
        label_right = float(current["x2"].max())

    elif "x_right" in current.columns:
        label_right = float(current["x_right"].max())

    else:
        label_right = cleaned_chart.shape[1] * 0.35

    label_right = int(
        max(1, min(round(label_right) + PROVINCE_RECOVERY_X_PADDING, cleaned_chart.shape[1]))
    )

    sorted_centers = sorted(province_centers)

    gaps = [b - a for a, b in pairwise(sorted_centers) if b > a]

    if gaps:
        half_height = float(np.median(gaps)) * 0.45
    else:
        half_height = 9.0

    half_height = round(
        max(PROVINCE_RECOVERY_HALF_HEIGHT_MIN, min(half_height, PROVINCE_RECOVERY_HALF_HEIGHT_MAX))
    )

    reader = get_ocr_reader()

    recovered_rows = []

    numbers = number_df.sort_values("center_y").reset_index(drop=True)

    for _, number in numbers.iterrows():
        number_y = float(number["center_y"])

        if province_centers:
            nearest_distance = min(abs(number_y - center_y) for center_y in province_centers)
        else:
            nearest_distance = float("inf")

        if nearest_distance <= PROVINCE_RECOVERY_MAX_Y_DISTANCE:
            continue

        crop_y1 = max(0, round(number_y - half_height))

        crop_y2 = min(cleaned_chart.shape[0], round(number_y + half_height))

        if crop_y2 <= crop_y1:
            continue

        row_crop = cleaned_chart[crop_y1:crop_y2, 0:label_right].copy()

        if row_crop.size == 0:
            continue

        gray = cv2.cvtColor(row_crop, cv2.COLOR_RGB2GRAY)

        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        _, threshold = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        row_images = {"row_original": row_crop, "row_gray": gray, "row_threshold": threshold}

        candidates = []

        for preprocess_name, image in row_images.items():
            detections = reader.readtext(image, detail=1, paragraph=False)

            scale = 1 if preprocess_name == "row_original" else 3

            for detection in detections:
                parsed = parse_easyocr_detection(detection)

                if parsed is None:
                    continue

                bbox, text, confidence = parsed

                if confidence < OCR_MIN_CONFIDENCE:
                    continue

                province, score = match_province(
                    text, threshold=(PROVINCE_RECOVERY_MIN_SCORE), margin=0.0
                )

                if province is None:
                    continue

                if province in existing_provinces:
                    continue

                x1 = float(bbox[:, 0].min()) / scale

                y1 = crop_y1 + float(bbox[:, 1].min()) / scale

                x2 = float(bbox[:, 0].max()) / scale

                y2 = crop_y1 + float(bbox[:, 1].max()) / scale

                candidates.append(
                    {
                        "province": province,
                        "ocr_text": str(text),
                        "confidence": float(confidence),
                        "province_score": float(score),
                        "center_x": (x1 + x2) / 2,
                        "center_y": (y1 + y2) / 2,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "x_left": x1,
                        "x_right": x2,
                        "y_top": y1,
                        "y_bottom": y2,
                        "preprocess": preprocess_name,
                    }
                )

        if not candidates:
            continue

        candidates = [
            candidate
            for candidate in candidates
            if abs(float(candidate["center_y"]) - number_y) <= half_height + 2
        ]

        if not candidates:
            continue

        candidates.sort(key=lambda item: (item["province_score"], item["confidence"]), reverse=True)

        best = candidates[0]

        province = best["province"]

        if province in existing_provinces:
            continue

        recovered_rows.append(best)

        existing_provinces.add(province)

        province_centers.append(float(best["center_y"]))

        if len(recovered_rows) >= max_recoveries:
            break

    if not recovered_rows:
        return current

    recovered_df = pd.DataFrame(recovered_rows)

    print(f"      -> Province row recovery: {len(recovered_df)}")

    return (
        pd.concat([current, recovered_df], ignore_index=True)
        .sort_values("center_y")
        .reset_index(drop=True)
    )


def remove_number_noise(binary):
    inverted = 255 - binary

    (num_labels, labels, stats, _) = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    cleaned = np.zeros_like(inverted)

    h_img, w_img = inverted.shape

    kept = 0

    for label_id in range(1, num_labels):
        w = stats[label_id, cv2.CC_STAT_WIDTH]

        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        if w <= 1 and h <= 2:
            continue

        if w > 0.40 * w_img:
            continue

        if h > 0.50 * h_img:
            continue

        cleaned[labels == label_id] = 255

        kept += 1

    return (255 - cleaned, kept)


def thicken_digits(binary):
    inverted = 255 - binary

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATION_KERNEL_SIZE)

    thick = cv2.dilate(inverted, kernel, iterations=DILATION_ITERATIONS)

    return 255 - thick


def build_number_images(cleaned_chart):
    number_gray = cv2.cvtColor(cleaned_chart, cv2.COLOR_RGB2GRAY)

    number_scaled = cv2.resize(
        number_gray,
        None,
        fx=NUMBER_RESIZE_SCALE,
        fy=NUMBER_RESIZE_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )

    _, threshold_40 = cv2.threshold(number_scaled, NUMBER_THRESHOLD, 255, cv2.THRESH_BINARY)

    if ENABLE_COMPONENT_FILTER:
        number_clean, kept = remove_number_noise(threshold_40)

    else:
        number_clean = threshold_40.copy()

        kept = None

    number_thick = thicken_digits(number_clean)

    return {
        "gray": number_gray,
        "scaled": number_scaled,
        "threshold_40": threshold_40,
        "clean": number_clean,
        "thickened": number_thick,
        "components_kept": kept,
    }


def detect_raw_number_candidates(number_thick) -> pd.DataFrame:
    reader = get_ocr_reader()

    detections = reader.readtext(number_thick, detail=1, paragraph=False, allowlist="0123456789")

    rows = []

    for raw_id, detection in enumerate(detections):
        parsed = parse_easyocr_detection(detection)

        if parsed is None:
            continue

        bbox, text, confidence = parsed

        if confidence < OCR_MIN_CONFIDENCE:
            continue

        digits = re.sub(r"\D", "", str(text))

        if not digits:
            continue

        x1 = float(bbox[:, 0].min()) / NUMBER_RESIZE_SCALE

        y1 = float(bbox[:, 1].min()) / NUMBER_RESIZE_SCALE

        x2 = float(bbox[:, 0].max()) / NUMBER_RESIZE_SCALE

        y2 = float(bbox[:, 1].max()) / NUMBER_RESIZE_SCALE

        rows.append(
            {
                "raw_id": raw_id,
                "ocr_text": str(text),
                "confidence": float(confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "raw_id",
                "ocr_text",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
            ]
        )

    return pd.DataFrame(rows)


def extract_components(binary_crop):
    if binary_crop is None or binary_crop.size == 0:
        return []

    foreground = 255 - binary_crop

    (num_labels, _labels, stats, centroids) = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )

    components = []

    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])

        y = int(stats[label_id, cv2.CC_STAT_TOP])

        w = int(stats[label_id, cv2.CC_STAT_WIDTH])

        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        area = int(stats[label_id, cv2.CC_STAT_AREA])

        cx = float(centroids[label_id, 0])

        cy = float(centroids[label_id, 1])

        if area < MIN_COMPONENT_AREA:
            continue

        if h < MIN_COMPONENT_HEIGHT:
            continue

        components.append({"x": x, "y": y, "w": w, "h": h, "area": area, "cx": cx, "cy": cy})

    return components


def group_components_by_y(components):
    if not components:
        return []

    components = sorted(components, key=lambda c: c["cy"])

    groups = []

    for component in components:
        best_group = None
        best_distance = None

        component_y1 = component["y"]

        component_y2 = component["y"] + component["h"]

        component_cy = component["cy"]

        for group in groups:
            group_y1 = min(c["y"] for c in group)

            group_y2 = max(c["y"] + c["h"] for c in group)

            group_cy = np.mean([c["cy"] for c in group])

            vertical_overlap = min(component_y2, group_y2) - max(component_y1, group_y1)

            distance = abs(component_cy - group_cy)

            same_row = vertical_overlap > 0 or distance <= ROW_SPLIT_Y_GAP

            if not same_row:
                continue

            if best_distance is None or distance < best_distance:
                best_group = group
                best_distance = distance

        if best_group is None:
            groups.append([component])

        else:
            best_group.append(component)

    groups.sort(key=lambda group: np.mean([c["cy"] for c in group]))

    return groups


def split_horizontal_number_groups(y_group):
    if not y_group:
        return []

    sorted_components = sorted(y_group, key=lambda c: c["x"])

    horizontal_groups = [[sorted_components[0]]]

    for component in sorted_components[1:]:
        current_group = horizontal_groups[-1]

        previous = current_group[-1]

        previous_right = previous["x"] + previous["w"]

        gap = component["x"] - previous_right

        if gap <= NUMBER_HORIZONTAL_GAP:
            current_group.append(component)

        else:
            horizontal_groups.append([component])

    return horizontal_groups


def split_raw_number_box(raw_row, number_thick):
    h, w = number_thick.shape

    x1 = max(0, int(raw_row["x1"] * NUMBER_RESIZE_SCALE))

    y1 = max(0, int(raw_row["y1"] * NUMBER_RESIZE_SCALE))

    x2 = min(w, int(raw_row["x2"] * NUMBER_RESIZE_SCALE))

    y2 = min(h, int(raw_row["y2"] * NUMBER_RESIZE_SCALE))

    if x2 <= x1 or y2 <= y1:
        return []

    binary_crop = number_thick[y1:y2, x1:x2].copy()

    components = extract_components(binary_crop)

    if not components:
        return []

    y_groups = group_components_by_y(components)

    result = []

    for y_group in y_groups:
        horizontal_groups = split_horizontal_number_groups(y_group)

        for x_group in horizontal_groups:
            local_x1 = min(c["x"] for c in x_group)

            local_y1 = min(c["y"] for c in x_group)

            local_x2 = max(c["x"] + c["w"] for c in x_group)

            local_y2 = max(c["y"] + c["h"] for c in x_group)

            result.append(
                {
                    "raw_id": int(raw_row["raw_id"]),
                    "split_id": len(result),
                    "x1": (x1 + local_x1) / NUMBER_RESIZE_SCALE,
                    "y1": (y1 + local_y1) / NUMBER_RESIZE_SCALE,
                    "x2": (x1 + local_x2) / NUMBER_RESIZE_SCALE,
                    "y2": (y1 + local_y2) / NUMBER_RESIZE_SCALE,
                    "component_count": len(x_group),
                }
            )

    return result


def ocr_split_box(image, box):
    reader = get_ocr_reader()

    h, w = image.shape

    x1 = max(0, int(box["x1"] * NUMBER_RESIZE_SCALE))

    y1 = max(0, int(box["y1"] * NUMBER_RESIZE_SCALE))

    x2 = min(w, int(box["x2"] * NUMBER_RESIZE_SCALE))

    y2 = min(h, int(box["y2"] * NUMBER_RESIZE_SCALE))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return None

    pad = 10

    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    detections = reader.readtext(crop, detail=1, paragraph=False, allowlist="0123456789")

    if not detections:
        return None

    candidates = []

    for detection in detections:
        parsed = parse_easyocr_detection(detection)

        if parsed is None:
            continue

        bbox, text, confidence = parsed

        if confidence < OCR_MIN_CONFIDENCE:
            continue

        text = re.sub(r"\D", "", str(text))

        if not text:
            continue

        try:
            value = int(text)
        except ValueError:
            continue

        local_x1 = float(bbox[:, 0].min()) - pad

        local_y1 = float(bbox[:, 1].min()) - pad

        local_x2 = float(bbox[:, 0].max()) - pad

        local_y2 = float(bbox[:, 1].max()) - pad

        global_x1 = (x1 + local_x1) / NUMBER_RESIZE_SCALE

        global_y1 = (y1 + local_y1) / NUMBER_RESIZE_SCALE

        global_x2 = (x1 + local_x2) / NUMBER_RESIZE_SCALE

        global_y2 = (y1 + local_y2) / NUMBER_RESIZE_SCALE

        candidates.append(
            {
                "value": value,
                "ocr_text": str(text),
                "confidence": float(confidence),
                "x1": global_x1,
                "y1": global_y1,
                "x2": global_x2,
                "y2": global_y2,
                "center_x": (global_x1 + global_x2) / 2,
                "center_y": (global_y1 + global_y2) / 2,
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["confidence"], reverse=True)

    return candidates[0]


def extract_number_candidates(cleaned_chart) -> pd.DataFrame:
    images = build_number_images(cleaned_chart)

    number_scaled = images["scaled"]

    number_thick = images["thickened"]

    raw_df = detect_raw_number_candidates(number_thick)

    print(f"      -> Raw number boxes: {len(raw_df)}")

    empty_columns = [
        "raw_id",
        "split_id",
        "value",
        "ocr_text",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "preprocess",
    ]

    if raw_df.empty:
        return pd.DataFrame(columns=empty_columns)

    all_split_boxes = []

    for _, raw_row in raw_df.iterrows():
        split_boxes = split_raw_number_box(raw_row, number_thick)

        all_split_boxes.extend(split_boxes)

    print(f"      -> Split number boxes: {len(all_split_boxes)}")

    if not all_split_boxes:
        return pd.DataFrame(columns=empty_columns)

    results = []

    for index, box in enumerate(all_split_boxes):
        result = ocr_split_box(number_scaled, box)

        if result is None:
            continue

        results.append(
            {
                "raw_id": box["raw_id"],
                "split_id": index,
                "value": result["value"],
                "ocr_text": result["ocr_text"],
                "confidence": result["confidence"],
                "x1": result["x1"],
                "y1": result["y1"],
                "x2": result["x2"],
                "y2": result["y2"],
                "center_x": result["center_x"],
                "center_y": result["center_y"],
                "preprocess": ("threshold_40_split"),
            }
        )

    print(f"      -> Final number OCR: {len(results)}")

    if not results:
        return pd.DataFrame(columns=empty_columns)

    return pd.DataFrame(results).sort_values(["center_y", "center_x"]).reset_index(drop=True)


def is_text_parser_result_valid(rows) -> bool:
    if rows is None:
        return False

    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return False

        if "province" not in rows.columns:
            return False

        if "cases" not in rows.columns:
            return False

        valid_rows = rows[rows["province"].notna() & rows["cases"].notna()]

        province_count = valid_rows["province"].nunique()

    else:
        valid_rows = [
            row for row in rows if row.get("province") is not None and row.get("cases") is not None
        ]

        province_count = len({row["province"] for row in valid_rows})

    if province_count < TEXT_PARSER_MIN_PROVINCES:
        return False

    if not TEXT_PARSER_REQUIRE_COMPLETE:
        return True

    return province_count >= EXPECTED_PROVINCE_COUNT


def _image_bbox_to_pdf_bbox(bbox) -> tuple[float, float, float, float]:
    return tuple(float(value) / RENDER_SCALE for value in bbox)


def _masked_regions_to_pdf_bboxes(
    masked_regions, crop_bbox
) -> list[tuple[float, float, float, float]]:
    crop_x1, crop_y1, _, _ = crop_bbox

    pdf_bboxes = []

    for region in masked_regions:
        local_bbox = region.get("bbox")

        if local_bbox is None:
            continue

        lx1, ly1, lx2, ly2 = map(float, local_bbox)

        image_bbox = (crop_x1 + lx1, crop_y1 + ly1, crop_x1 + lx2, crop_y1 + ly2)

        pdf_bboxes.append(_image_bbox_to_pdf_bbox(image_bbox))

    return pdf_bboxes


def extract_dengue_ocr_table(
    pdf_path: Path,
    page_number: int | None = None,
    target_chart_index: int = (DEFAULT_TARGET_CHART_INDEX),
) -> list[dict]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    if page_number is None:
        dengue_pages = find_dengue_pages(pdf_path)

        if not dengue_pages:
            return []

        page_number = int(dengue_pages[0])

    rendered_page = render_page(pdf_path, page_number)

    layout_df = detect_layout(rendered_page)

    if layout_df.empty:
        return []

    target_chart = select_target_chart(layout_df, target_chart_index)

    (chart_crop, chart_bbox, crop_bbox) = crop_target_chart(rendered_page, target_chart)

    (cleaned_chart, masked_regions) = mask_non_chart_regions(
        chart_crop, layout_df, chart_bbox, crop_bbox
    )

    pdf_chart_bbox = _image_bbox_to_pdf_bbox(chart_bbox)

    excluded_bboxes = _masked_regions_to_pdf_bboxes(masked_regions, crop_bbox)

    try:
        text_rows = parse_dengue_page(
            pdf_path, page_number, chart_bbox=pdf_chart_bbox, excluded_bboxes=excluded_bboxes
        )

    except Exception as exc:
        print(f"      -> Text parser error: {type(exc).__name__}: {exc}")

        text_rows = []

    if is_text_parser_result_valid(text_rows):
        print("      -> Extraction method: PDF text layer")

        if isinstance(text_rows, pd.DataFrame):
            return text_rows.to_dict(orient="records")

        return text_rows

    print("      -> Text layer incomplete; fallback to OCR")

    province_df = extract_province_candidates(cleaned_chart)

    print(f"      -> Province detected: {len(province_df)}")

    number_df = extract_number_candidates(cleaned_chart)

    province_df = recover_missing_province_candidates(cleaned_chart, province_df, number_df)

    print(f"      -> Province detected after recovery: {len(province_df)}")

    rows = build_rows_from_candidates(province_df, number_df)

    for row in rows:
        row["ocr_page"] = page_number
        row["chart_bbox"] = chart_bbox
        row["masked_region_count"] = len(masked_regions)

    print(f"      -> OCR mapped rows: {len(rows)}")

    return rows
