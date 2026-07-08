import struct
from pathlib import Path
import tempfile
from app import geo_utils


def u16(data, off):
    return struct.unpack('<H', data[off:off+2])[0]


def u32(data, off):
    return struct.unpack('<I', data[off:off+4])[0]


def f64(data, off):
    return struct.unpack('<d', data[off:off+8])[0]


data = bytearray()
data.extend(b'II')
data.extend(struct.pack('<H', 42))
data.extend(struct.pack('<I', 8))
data.extend(struct.pack('<H', 2))
data.extend(struct.pack('<HHII', 33922, 12, 6, 64))
data.extend(struct.pack('<HHII', 33550, 12, 3, 96))
data.extend(struct.pack('<I', 0))
data.extend(b'\x00' * (64 - len(data)))
data.extend(struct.pack('<d', 1.0))
data.extend(struct.pack('<d', 2.0))
data.extend(struct.pack('<d', 0.0))
data.extend(struct.pack('<d', 10.0))
data.extend(struct.pack('<d', 20.0))
data.extend(struct.pack('<d', 0.0))
data.extend(b'\x00' * (96 - len(data)))
data.extend(struct.pack('<d', 0.1))
data.extend(struct.pack('<d', 0.2))
data.extend(struct.pack('<d', 1.0))
p = Path(tempfile.gettempdir()) / 'geo_test.tif'
p.write_bytes(data)
print('file exists', p.exists(), 'size', p.stat().st_size)
print('ifd', u32(data, 4))
print('n entries', u16(data, 8))
for i in range(2):
    entry = 10 + i * 12
    print('entry', i, entry, struct.unpack('<HHII', data[entry:entry+12]))
print('tiepoint values', [f64(data, 64 + k * 8) for k in range(6)])
print('pixel scale values', [f64(data, 96 + k * 8) for k in range(3)])
print('parse', geo_utils.parse_geotiff_transform(str(p)))
