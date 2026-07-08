"""v7.6-ROADBLOCK post-processing gauntlet.

Each function below implements one of the 5 rejection layers from the
master plan §3 / §8. They operate on individual contours extracted from
the binary YOLO mask and return True if the contour is REJECTED.

Pipeline order (caller responsibility):
    1. large_blob_integrity_check    (> 50 m²)
    2. road_blocker                  (aspect ratio, rect fill, min width)
    3. shadow_rejection              (mean intensity < 65)
    4. road_texture_rejection        (std < 12 for areas > 80 m²)
    5. vegetation_color_rejection    (HSV green range)
    + boundary_straightening         (approxPolyDP, applied before checks)
    + internal_structure_exclusion   (eroded interior edge ratio ≥ 15%)
"""
from __future__ import annotations

import math
import numpy as np
import cv2

from .config import INFERENCE


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def contour_metrics(contour: np.ndarray, m_per_px: float) -> dict:
    """Compute solidity, extent, compactness, area (m²), bounding rect."""
    area_px = cv2.contourArea(contour)
    peri_px = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h

    solidity = (area_px / hull_area) if hull_area > 0 else 0
    extent = (area_px / rect_area) if rect_area > 0 else 0
    compactness = (peri_px ** 2 / area_px) if area_px > 0 else float("inf")
    aspect = (w / h) if h > 0 else 0

    return {
        "area_px": area_px,
        "area_m2": area_px * (m_per_px ** 2),
        "perimeter_px": peri_px,
        "solidity": solidity,
        "extent": extent,
        "compactness": compactness,
        "aspect_ratio": aspect,
        "bbox": (x, y, w, h),
        "width_m": w * m_per_px,
        "height_m": h * m_per_px,
    }


# ---------------------------------------------------------------------
# Layer 1: Large-blob integrity (>50 m²)
# ---------------------------------------------------------------------
def large_blob_integrity_check(metrics: dict) -> bool:
    """Reject blobs > 50 m² that fail solidity/extent/compactness thresholds."""
    if metrics["area_m2"] < INFERENCE["min_blob_area_m2"]:
        return False    # small blob — pass through (handled by other layers)
    if metrics["solidity"] < INFERENCE["solidity_min"]:
        return True
    if metrics["extent"] < INFERENCE["extent_min"]:
        return True
    if metrics["compactness"] > INFERENCE["compactness_max"]:
        return True
    return False


# ---------------------------------------------------------------------
# Layer 2: Road Blocker
# ---------------------------------------------------------------------
def rectangular_fill_score(contour: np.ndarray, metrics: dict) -> float:
    """How well does the contour fill its bounding rect? 1.0 = perfect rect."""
    x, y, w, h = metrics["bbox"]
    mask = np.zeros((h, w), dtype="uint8")
    cv2.drawContours(mask, [contour - np.array([x, y])], -1, 255, -1)
    filled = cv2.countNonZero(mask)
    return filled / (w * h) if (w * h) > 0 else 0


def road_blocker(contour: np.ndarray, metrics: dict) -> bool:
    """Reject thin, very rectangular shapes — likely roads, not roofs."""
    if metrics["aspect_ratio"] > INFERENCE["aspect_ratio_max"]:
        return True
    if metrics["width_m"] < INFERENCE["min_width_m"]:
        return True
    fill = rectangular_fill_score(contour, metrics)
    if fill >= INFERENCE["rect_fill_min"]:
        return True
    return False


# ---------------------------------------------------------------------
# Layer 3: Shadow rejection (mean grayscale intensity < 65)
# ---------------------------------------------------------------------
def shadow_rejection(gray_image: np.ndarray, contour: np.ndarray) -> bool:
    """Reject very dark regions — likely shadows, not roofs."""
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_val = cv2.mean(gray_image, mask=mask)[0]
    return mean_val < INFERENCE["shadow_intensity_max"]


# ---------------------------------------------------------------------
# Layer 4: Road texture rejection (areas > 80 m², std < 12)
# ---------------------------------------------------------------------
def road_texture_rejection(gray_image: np.ndarray, contour: np.ndarray,
                            metrics: dict) -> bool:
    """Reject large flat regions — likely roads/parking lots."""
    if metrics["area_m2"] < INFERENCE["road_area_thresh_m2"]:
        return False
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    _, stddev = cv2.meanStdDev(gray_image, mask=mask)
    return float(stddev[0, 0]) < INFERENCE["road_std_max"]


# ---------------------------------------------------------------------
# Layer 5: Vegetation color rejection (HSV green range)
# ---------------------------------------------------------------------
def vegetation_color_rejection(hsv_image: np.ndarray, contour: np.ndarray) -> bool:
    """Reject green-dominant regions — likely vegetation, not roofs."""
    mask = np.zeros(hsv_image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    green_lo = np.array([35, 40, 40])
    green_hi = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, green_lo, green_hi)
    green_in_contour = cv2.countNonZero(cv2.bitwise_and(green_mask, mask))
    total_in_contour = cv2.countNonZero(mask)
    if total_in_contour == 0:
        return False
    return (green_in_contour / total_in_contour) > 0.5


# ---------------------------------------------------------------------
# Boundary straightening
# ---------------------------------------------------------------------
def boundary_straightening(contour: np.ndarray) -> np.ndarray:
    """approxPolyDP with 1.2% of perimeter tolerance."""
    peri = cv2.arcLength(contour, True)
    eps = INFERENCE["approx_dp_ratio"] * peri
    return cv2.approxPolyDP(contour, eps, True)


# ---------------------------------------------------------------------
# Internal structure exclusion
# ---------------------------------------------------------------------
def internal_structure_exclusion(contour: np.ndarray,
                                  gray_image: np.ndarray) -> bool:
    """Reject contours whose interior has lots of edges — likely contains
    internal subdivisions (walls), not a single roof.

    The intended check (per master plan §3.8) is:
      - Mask the gray image inside the contour
      - Run edge detection on the interior
      - If edge-pixel ratio ≥ 15% → reject (it's subdivided, not a single roof)

    A previous version of this function incorrectly used XOR(canvas, eroded)
    which computes the contour BOUNDARY ratio, not interior edges. That
    rejected all small rectangles. This version uses Canny inside the mask.
    """
    x, y, w, h = cv2.boundingRect(contour)
    if w < 10 or h < 10:
        return False
    # Build a mask of just the contour interior (no boundary)
    mask = np.zeros(gray_image.shape, dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    interior_mask = cv2.erode(mask, np.ones((3, 3), "uint8"), iterations=1)
    if cv2.countNonZero(interior_mask) == 0:
        return False
    # Canny edges inside the interior
    edges = cv2.Canny(gray_image, 50, 150)
    edges_in = cv2.bitwise_and(edges, interior_mask)
    edge_ratio = cv2.countNonZero(edges_in) / cv2.countNonZero(interior_mask)
    return edge_ratio >= INFERENCE["interior_edge_ratio_max"]


# ---------------------------------------------------------------------
# Full gauntlet
# ---------------------------------------------------------------------
def run_gauntlet(contour: np.ndarray,
                 gray_image: np.ndarray,
                 hsv_image: np.ndarray,
                 m_per_px: float) -> tuple[bool, str, dict]:
    """Apply all 5 layers + boundary straightening + internal structure.

    Returns:
        (rejected: bool, reason: str, metrics: dict)
    """
    # Pre-step A: boundary straightening
    contour = boundary_straightening(contour)

    # Pre-step B: internal structure exclusion
    if internal_structure_exclusion(contour, gray_image):
        return True, "internal_structure", {}

    metrics = contour_metrics(contour, m_per_px)

    # Skip the gauntlet for very small blobs (<5 m²) — too small to evaluate
    if metrics["area_m2"] < 5.0:
        return False, "ok_small", metrics

    # Layer 1: large-blob integrity
    if large_blob_integrity_check(metrics):
        return True, "large_blob_integrity", metrics

    # Layer 2: road blocker — only applies to very elongated shapes
    if metrics["aspect_ratio"] > INFERENCE["aspect_ratio_max"]:
        return True, "road_blocker_aspect", metrics
    if metrics["width_m"] < INFERENCE["min_width_m"]:
        return True, "road_blocker_width", metrics
    # Note: We deliberately DON'T reject high rect_fill_score — real roofs
    # are often rectangular. Only flag very high aspect ratio + thin shapes.

    # Layer 3: shadow
    if shadow_rejection(gray_image, contour):
        return True, "shadow", metrics

    # Layer 4: road texture
    if road_texture_rejection(gray_image, contour, metrics):
        return True, "road_texture", metrics

    # Layer 5: vegetation color
    if vegetation_color_rejection(hsv_image, contour):
        return True, "vegetation_color", metrics

    return False, "ok", metrics


def filter_contours(contours: list[np.ndarray],
                    gray_image: np.ndarray,
                    hsv_image: np.ndarray,
                    m_per_px: float) -> list[tuple[np.ndarray, dict]]:
    """Apply the gauntlet to every contour; return survivors with metrics."""
    survivors = []
    for c in contours:
        if cv2.contourArea(c) < 32:
            continue
        rejected, _, metrics = run_gauntlet(c, gray_image, hsv_image, m_per_px)
        if not rejected:
            survivors.append((c, metrics))
    return survivors
