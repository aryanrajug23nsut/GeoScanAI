import struct
import tempfile
from pathlib import Path

from app import geo_utils


def test_parse_geotiff_transform_uses_tiepoint_offset():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.tif"
        data = bytearray()

        # Minimal TIFF header + IFD with two tags.
        data.extend(b"II")
        data.extend(struct.pack("<H", 42))
        data.extend(struct.pack("<I", 8))

        # IFD at offset 8
        data.extend(struct.pack("<H", 2))

        # Tag 33922: ModelTiepointTag (count 6, type 12/double)
        data.extend(struct.pack("<HHII", 33922, 12, 6, 64))

        # Tag 33550: ModelPixelScaleTag (count 3, type 12/double)
        # The tag payload starts after the first 6 doubles (48 bytes).
        data.extend(struct.pack("<HHII", 33550, 12, 3, 112))

        data.extend(struct.pack("<I", 0))  # next IFD offset

        # Pad to the first value offset.
        data.extend(b"\x00" * (64 - len(data)))

        # Values payload for the tiepoint tag.
        data.extend(struct.pack("<d", 1.0))
        data.extend(struct.pack("<d", 2.0))
        data.extend(struct.pack("<d", 0.0))
        data.extend(struct.pack("<d", 10.0))
        data.extend(struct.pack("<d", 20.0))
        data.extend(struct.pack("<d", 0.0))

        # Pad to the pixel scale offset.
        data.extend(b"\x00" * (96 - len(data)))

        data.extend(struct.pack("<d", 0.1))
        data.extend(struct.pack("<d", 0.2))
        data.extend(struct.pack("<d", 1.0))

        path.write_bytes(data)

        transform = geo_utils.parse_geotiff_transform(str(path))

        assert transform is not None
        lon, lat = transform.pixel_to_world(3, 4)
        assert round(lon, 7) == 10.2
        assert round(lat, 7) == 19.6
