"""Geo utility functions."""
import math


def haversine_meters(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points, in meters."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _dms_to_decimal(dms, ref) -> float:
    """Converts an EXIF GPS (degrees, minutes, seconds) rational triple
    to signed decimal degrees. `dms` is a 3-tuple of numbers (Pillow gives
    IFDRational objects, which behave like floats); `ref` is one of
    'N'/'S'/'E'/'W'."""
    degrees, minutes, seconds = (float(x) for x in dms)
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_gps_from_exif(pil_image) -> tuple[float, float] | tuple[None, None]:
    """Reads real GPS coordinates out of a photo's EXIF metadata, exactly
    the way a real smartphone-captured evidence photo would carry them
    (assuming location services were on when it was taken) — this is
    standard, well-established metadata, not something invented for this
    app. Returns (lat, lon) if present and parseable, else (None, None).

    Deliberately does NOT accept or fall back to manually-typed
    coordinates: a self-reported location isn't a location check — a
    fraudulent upload could just as easily "self-report" the correct
    site's coordinates, which defeats the entire point. If a photo has no
    GPS EXIF (common for downloaded/forwarded/screenshotted images, or a
    camera with location services off), the honest answer is that this
    photo's location cannot be automatically verified — not a fabricated
    coordinate."""
    try:
        exif = pil_image.getexif()
        if not exif:
            return None, None
        # GPS data lives in a nested IFD (tag 0x8825 / 34853), not the
        # flat top-level EXIF dict — Pillow >= 9.1 exposes it via get_ifd.
        from PIL import ExifTags
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        if not gps_ifd:
            return None, None

        lat_dms = gps_ifd.get(2)   # GPSLatitude
        lat_ref = gps_ifd.get(1)   # GPSLatitudeRef ('N'/'S')
        lon_dms = gps_ifd.get(4)   # GPSLongitude
        lon_ref = gps_ifd.get(3)   # GPSLongitudeRef ('E'/'W')
        if not (lat_dms and lat_ref and lon_dms and lon_ref):
            return None, None

        lat = _dms_to_decimal(lat_dms, lat_ref)
        lon = _dms_to_decimal(lon_dms, lon_ref)
        return lat, lon
    except Exception:
        # Any malformed/unexpected EXIF structure -> treat as "no GPS data"
        # rather than raising and breaking the upload flow.
        return None, None
