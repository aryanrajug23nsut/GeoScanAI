"""Geospatial utilities — the heart of polygon mapping.

This module answers the user's key question:
"How are polygons mapped on the basemap if only two lat/lon are generated?"

Answer: We DO NOT generate only two coordinates. Every detected object is
returned as a FULL polygon ring (5+ [lon, lat] points forming a closed
shape), stored as GeoJSON in the database, and rendered on Leaflet as
L.polygon. The two-number (lat, lon) you may see in the CSV is just the
CENTROID for quick reference — the actual geometry is the full ring.

Pipeline:
  1. Backend reads the uploaded .tif and (if GeoTIFF) extracts its CRS
     and affine transform → we know pixel (x, y) → world (lon, lat).
  2. YOLO + post-processing produces a binary mask. We extract contours
     with cv2.findContours → each contour is a list of pixel (x, y) points.
  3. pixel_to_world() converts every contour point to EPSG:4326 (lon, lat).
  4. build_geojson_feature() wraps the ring as a GeoJSON Polygon Feature.
  5. Frontend L.polygon(ring) draws it on the Leaflet map.

If the file is a plain .jpg/.png (no georeference), we fall back to a
synthetic transform anchored at the map's current center and the file's
EXIF DPI for scale — so polygons still render in a sensible location.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

# Use TYPE_CHECKING so type checkers see numpy/cv2 properly,
# while still allowing runtime import failures for optional dependencies.
if TYPE_CHECKING:
    import numpy as np
    import cv2

# Runtime guards for optional dependencies
try:
    import numpy as _np_runtime  # noqa: F401
    _HAS_NUMPY = True
except ImportError:
    _np_runtime = None
    _HAS_NUMPY = False

try:
    import cv2 as _cv2_runtime  # type: ignore
    _HAS_CV2 = True
except ImportError:
    _cv2_runtime = None
    _HAS_CV2 = False


# ---------------------------------------------------------------------
# GeoTIFF metadata extraction (lightweight — uses pure Python, no GDAL)
# ---------------------------------------------------------------------
@dataclass
class GeoTransform:
    """Maps pixel (col, row) → world (lon, lat) in EPSG:4326."""
    origin_lon: float
    origin_lat: float
    lon_per_px: float
    lat_per_px: float       # negative (image rows go south)
    crs: str = "EPSG:4326"

    def pixel_to_world(self, col: float, row: float) -> tuple[float, float]:
        lon = self.origin_lon + col * self.lon_per_px
        lat = self.origin_lat + row * self.lat_per_px
        return (lon, lat)

    def to_geojson_ring(self, width_px: int, height_px: int) -> list[list[float]]:
        """Return the 5-point closed ring of the image footprint in EPSG:4326."""
        tl = self.pixel_to_world(0, 0)
        tr = self.pixel_to_world(width_px, 0)
        br = self.pixel_to_world(width_px, height_px)
        bl = self.pixel_to_world(0, height_px)
        return [tl, tr, br, bl, tl]   # closed ring


def parse_geotiff_transform(file_path: str) -> GeoTransform | None:
    """Read GeoTIFF ModelTiepoint/ModelPixelScale tags without GDAL.

    Returns None if the file is not a GeoTIFF or has no georeference.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        if data[:2] != b"II" and data[:2] != b"MM":
            return None

        big_endian = data[:2] == b"MM"
        def u16(off): return struct.unpack(">H" if big_endian else "<H", data[off:off+2])[0]
        def u32(off): return struct.unpack(">I" if big_endian else "<I", data[off:off+4])[0]
        def f64(off): return struct.unpack(">d" if big_endian else "<d", data[off:off+8])[0]

        def read_values(off: int, typ: int, count: int) -> list[float]:
            if typ == 12:
                return [f64(off + k * 8) for k in range(count)]
            if typ == 3:
                return [u16(off + k * 2) for k in range(count)]
            if typ == 4:
                return [u32(off + k * 4) for k in range(count)]
            return []

        ifd_off = u32(4)
        n_entries = u16(ifd_off)

        tiepoint = None
        pixel_scale = None
        for i in range(n_entries):
            entry = ifd_off + 2 + i * 12
            tag = u16(entry)
            typ = u16(entry + 2)
            count = u32(entry + 4)
            value_offset = u32(entry + 8)
            value_bytes_offset = value_offset if value_offset + 8 <= len(data) else entry + 8
            if tag == 33922:   # ModelTiepointTag (raster->model tiepoint)
                # 6 doubles: i, j, k, x, y, z
                tiepoint = read_values(value_bytes_offset, typ, min(count, 6))
            elif tag == 33550:  # ModelPixelScaleTag
                # 3 doubles: sx, sy, sz
                pixel_scale = read_values(value_bytes_offset, typ, min(count, 3))

        if tiepoint and pixel_scale and len(tiepoint) >= 6 and len(pixel_scale) >= 2:
            i, j, _, x, y, _ = tiepoint[:6]
            lon_per_px = float(pixel_scale[0])
            lat_per_px = -float(pixel_scale[1])
            if lon_per_px == 0 or lat_per_px == 0:
                return None
            # Use the tiepoint offset so the polygon is anchored to the uploaded image's true location.
            origin_lon = x - i * lon_per_px
            origin_lat = y - j * lat_per_px
            return GeoTransform(
                origin_lon=origin_lon,
                origin_lat=origin_lat,
                lon_per_px=lon_per_px,
                lat_per_px=lat_per_px,
                crs="EPSG:4326"
            )
    except Exception:
        pass
    return None


def synthetic_transform(center_lat: float, center_lon: float,
                         width_px: int, height_px: int,
                         dpi: int = 96) -> GeoTransform:
    """Build a transform when the image has no georeference.

    Uses the cursor center and DPI-derived scale so polygons land in a
    sensible spot. The user can still pan/zoom to refine.
    """
    # 1 inch = 0.0254 m. Ground scale assumes satellite altitude ~ zoom-appropriate.
    meters_per_px = 0.0254 / dpi * 100.0  # heuristic: ~2.6 m/px at 96 DPI
    lon_per_px = meters_per_px / (111320.0 * math.cos(math.radians(center_lat)))
    lat_per_px = -meters_per_px / 110540.0

    # Anchor image center at (center_lon, center_lat)
    origin_lon = center_lon - (width_px / 2) * lon_per_px
    origin_lat = center_lat - (height_px / 2) * lat_per_px

    return GeoTransform(
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        lon_per_px=lon_per_px,
        lat_per_px=lat_per_px,
        crs="EPSG:4326"
    )


# ---------------------------------------------------------------------
# Polygon construction from binary mask contours
# ---------------------------------------------------------------------
def mask_to_polygons(mask, transform: GeoTransform,
                     min_area_px: int = 32) -> list[list[list[float]]]:
    """Convert a binary mask to a list of GeoJSON polygon rings (EPSG:4326).

    Each ring is a list of [lon, lat] points, closed (first == last).
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for mask_to_polygons but is not installed.")
    cv2 = _cv2_runtime  # local alias for readability

    contours, _ = cv2.findContours(
        mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rings = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        # Simplify contour to ~1.2% of perimeter (Boundary Straightening)
        peri = cv2.arcLength(c, True)
        eps = 0.012 * peri
        approx = cv2.approxPolyDP(c, eps, True)

        ring = []
        pts = approx.squeeze(1) if approx.ndim == 3 else approx
        for pt in pts:
            col, row = float(pt[0]), float(pt[1])
            lon, lat = transform.pixel_to_world(col, row)
            ring.append([round(lon, 7), round(lat, 7)])
        if len(ring) >= 4:
            ring.append(ring[0])    # close the ring
            rings.append(ring)
    return rings


def build_geojson_feature(ring: list[list[float]],
                          feature_type: str,
                          properties: dict) -> dict:
    """Wrap a ring as a GeoJSON Polygon Feature.

    `feature_type` is 'rooftop' | 'solar_panel' | any class id.
    """
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [ring]
        },
        "properties": {
            "type": feature_type,
            **properties
        }
    }


def ring_centroid(ring: list[list[float]]) -> tuple[float, float]:
    """Return (lat, lon) centroid of a [lon, lat] ring."""
    n = len(ring) - 1   # last == first
    if n < 3:
        return (0.0, 0.0)
    sum_lat = sum(p[1] for p in ring[:-1])
    sum_lon = sum(p[0] for p in ring[:-1])
    return (sum_lat / n, sum_lon / n)


def ring_area_sqm(ring: list[list[float]]) -> float:
    """Approximate polygon area in m² using the shoelace formula
    with the EPSG:4326 → equirectangular projection at the ring centroid.
    """
    if not ring or len(ring) < 4:
        return 0.0
    lat0, _ = ring_centroid(ring)
    m_per_deg_lat = 110540.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))

    s = 0.0
    for i in range(len(ring) - 1):
        lon1, lat1 = ring[i]
        lon2, lat2 = ring[i + 1]
        s += (lon1 * m_per_deg_lon * lat2 * m_per_deg_lat) - \
             (lon2 * m_per_deg_lon * lat1 * m_per_deg_lat)
    return abs(s) / 2.0


def ring_to_wkt(ring: list[list[float]]) -> str:
    """Convert [lon, lat] ring → WKT POLYGON string."""
    coords = ", ".join(f"{lon} {lat}" for lon, lat in ring)
    return f"POLYGON(({coords}))"


# ---------------------------------------------------------------------
# KML / CSV export helpers
# ---------------------------------------------------------------------
def features_to_kml(features: list[dict]) -> str:
    placemarks = []
    for f in features:
        ring = f["geometry"]["coordinates"][0]
        kml_coords = " ".join(f"{lon},{lat},0" for lon, lat in ring)
        props = f.get("properties", {})
        placemarks.append(f"""
        <Placemark>
          <name>{props.get('type', 'feature')}</name>
          <ExtendedData>
            <Data name="area_m2"><value>{props.get('area_m2', 0)}</value></Data>
            <Data name="confidence"><value>{props.get('confidence', 0)}</value></Data>
            <Data name="model"><value>{props.get('model', '')}</value></Data>
          </ExtendedData>
          <Polygon><outerBoundaryIs><LinearRing><coordinates>{kml_coords}</coordinates></LinearRing></outerBoundaryIs></Polygon>
        </Placemark>""")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document><name>GeoScan.AI Detections</name>{''.join(placemarks)}
  </Document>
</kml>"""


def features_to_csv(features: list[dict]) -> str:
    import csv, io
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["feature_id", "type", "model", "area_m2", "confidence",
                "centroid_lat", "centroid_lon", "polygon_wkt"])
    for i, f in enumerate(features, 1):
        ring = f["geometry"]["coordinates"][0]
        props = f.get("properties", {})
        lat, lon = ring_centroid(ring)
        w.writerow([i, props.get("type", ""), props.get("model", ""),
                    round(props.get("area_m2", 0), 2),
                    round(props.get("confidence", 0), 3),
                    round(lat, 6), round(lon, 6),
                    ring_to_wkt(ring)])
    return out.getvalue()


# ---------------------------------------------------------------------
# Rasterio-based GeoTIFF reader (matches Colab v8.9 logic)
# ---------------------------------------------------------------------
def read_geotiff_rasterio(
    file_path: str,
) -> tuple[Any, "GeoTransform | None", Any]:
    """Read a GeoTIFF using rasterio. Returns (image_bgr, GeoTransform, crs).

    Handles:
      - Multi-band → 3-band RGB
      - 16-bit → 8-bit (2%-98% stretch like Colab)
      - Any CRS → returns GeoTransform that converts pixels to EPSG:4326 lat/lon
    """
    try:
        import rasterio  # type: ignore[import-not-found]
        from rasterio.warp import transform_bounds  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "rasterio is required to read GeoTIFFs with CRS metadata. "
            "Install it with `pip install rasterio`."
        ) from e

    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required by read_geotiff_rasterio.")
    cv2 = _cv2_runtime  # local alias to prevent NameError

    import numpy as np  # local; safe to assume present when rasterio is

    with rasterio.open(file_path) as src:
        crs = src.crs

        # Read bands
        bands = min(src.count, 3)
        img_rgb = np.transpose(src.read(list(range(1, bands + 1))), (1, 2, 0))
        if bands == 1:
            img_rgb = np.repeat(img_rgb, 3, axis=-1)

        # Convert to 8-bit if needed (same logic as Colab)
        if img_rgb.dtype != np.uint8:
            img_rgb = img_rgb.astype(np.float32)
            lo, hi = np.percentile(img_rgb, 2), np.percentile(img_rgb, 98)
            if hi > lo:
                img_rgb = np.clip((img_rgb - lo) / (hi - lo), 0, 1) * 255.0
            else:
                img_rgb = np.clip(img_rgb, 0, 255)
            img_rgb = img_rgb.astype(np.uint8)

        # RGB → BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Build GeoTransform that outputs EPSG:4326 lat/lon
        if crs is not None:
            # Get bounds in EPSG:4326
            try:
                west, south, east, north = transform_bounds(
                    crs, "EPSG:4326",
                    src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top
                )
                h, w = img_bgr.shape[:2]
                transform = GeoTransform(
                    origin_lon=west,
                    origin_lat=north,  # top-left is north
                    lon_per_px=(east - west) / w,
                    lat_per_px=-(north - south) / h,  # negative (rows go south)
                    crs="EPSG:4326"
                )
                return img_bgr, transform, crs
            except Exception as e:
                print(f"[geo_utils] CRS transform failed: {e}")

        return img_bgr, None, crs