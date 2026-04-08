"""
PoC: load sample Bangalore One centres, geocode user address via Nominatim, pick nearest by haversine.

No API key. Nominatim requires a descriptive User-Agent and conservative use.
"""

from __future__ import annotations

import csv
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

from graph_knowledge import DATA_DIR

DEFAULT_CENTRES_CSV = os.path.join(DATA_DIR, "bangalore_one_centres.csv")
EARTH_KM = 6371.0
USER_AGENT = "KarnatakaSchemesPoC/1.0 (local demo; contact: self-hosted)"


@dataclass(frozen=True)
class Centre:
    name: str
    area: str
    address: str
    lat: float
    lon: float
    notes: str


def load_centres(csv_path: str = DEFAULT_CENTRES_CSV) -> List[Centre]:
    path = csv_path
    if not os.path.isfile(path):
        return []
    out: List[Centre] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out.append(
                    Centre(
                        name=(row.get("name") or "").strip(),
                        area=(row.get("area") or "").strip(),
                        address=(row.get("address") or "").strip(),
                        lat=float(row["lat"]),
                        lon=float(row["lon"]),
                        notes=(row.get("notes") or "").strip(),
                    )
                )
            except (KeyError, ValueError):
                continue
    return out


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def nominatim_geocode(query: str, timeout_s: float = 12.0) -> Optional[Tuple[float, float]]:
    q = query.strip()
    if not q:
        return None
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": q, "format": "json", "limit": 1}
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    if not data:
        return None
    try:
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        return lat, lon
    except (KeyError, ValueError, TypeError):
        return None


def build_geocode_query(
    *,
    line1: str,
    pincode: str,
    city: str = "Bengaluru",
    state: str = "Karnataka",
    country: str = "India",
) -> str:
    parts = [p.strip() for p in (line1, pincode, city, state, country) if p and str(p).strip()]
    return ", ".join(parts)


def nearest_centre(lat: float, lon: float, centres: List[Centre]) -> Optional[Tuple[Centre, float]]:
    if not centres:
        return None
    best: Optional[Tuple[Centre, float]] = None
    for c in centres:
        d = haversine_km(lat, lon, c.lat, c.lon)
        if best is None or d < best[1]:
            best = (c, d)
    return best
