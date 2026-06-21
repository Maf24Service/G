import math

from .detector import _Detection
from .schemas import GeoPrediction


def _offset_coords(
    lat: float, lon: float, north_m: float, east_m: float
) -> tuple[float, float]:
    """Offset a lat/lon point by a number of meters north and east."""
    earth_radius = 6_378_137.0
    d_lat = north_m / earth_radius
    d_lon = east_m / (earth_radius * math.cos(math.radians(lat)))
    return lat + math.degrees(d_lat), lon + math.degrees(d_lon)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def predict_geo(
    latitude: float,
    longitude: float,
    accuracy: float,
    detections: list[_Detection],
    image_width: int,
    image_height: int,
) -> GeoPrediction:
    """Derive a navigation hint from detected landmarks.

    Uses the strongest landmark's horizontal position relative to the frame
    center to suggest a bearing, and projects a target point a short distance
    ahead. This is a deterministic heuristic meant as a working stand-in for a
    learned geolocation model.
    """
    if not detections:
        return GeoPrediction(
            action_required=(
                "No landmarks detected. Hold position and rescan the area."
            )
        )

    strongest = max(detections, key=lambda d: d.confidence)
    x1, _, x2, _ = strongest.box_pixels
    center_x = (x1 + x2) / 2.0
    frame_center = image_width / 2.0
    # Normalized horizontal offset in [-1, 1]; positive means landmark is right.
    offset = (center_x - frame_center) / frame_center

    if offset < -0.2:
        direction = "Turn left and advance toward the landmark."
        bearing_deg = -45.0
    elif offset > 0.2:
        direction = "Turn right and advance toward the landmark."
        bearing_deg = 45.0
    else:
        direction = "Move straight ahead toward the landmark."
        bearing_deg = 0.0

    # Project a target ~25m ahead along the suggested bearing.
    distance = 25.0
    north = distance * math.cos(math.radians(bearing_deg))
    east = distance * math.sin(math.radians(bearing_deg))
    pred_lat, pred_lon = _offset_coords(latitude, longitude, north, east)

    return GeoPrediction(
        action_required=f"{direction} (GPS accuracy ~{accuracy:.0f} m)",
        predicted_lat=pred_lat,
        predicted_lon=pred_lon,
        distance_meters=round(
            _haversine_m(latitude, longitude, pred_lat, pred_lon), 1
        ),
    )
