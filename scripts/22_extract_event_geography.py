"""
Phase 18: Event Geography & Travel Features
===========================================
Reads cached USCF crosstable HTML, extracts the **event-summary header**
fields (Location, Event Date, Sponsoring Affiliate, Stats, Chief TD),
geocodes events offline using a curated city table + US state-centroid
fallback (no live API), and joins distances into the player profile.

Outputs:
  data/processed/event_metadata.csv     — one row per event
  data/processed/player_travel_features.csv — one row per focal player
  data/processed/player_profiles.csv    — UPDATED with travel columns

Design constraints:
  * NO network calls / live geocoding.
  * NO re-scraping.  Reads only files already in `data/raw/html_cache/`.
  * If a feature cannot be computed (e.g. international event with no
    coordinates), the row gets a clear `event_location_confidence`
    label rather than silent imputation.
  * Travel distance is always presented as **approximate** because we
    only know city/state, not exact venue GPS.

Caveat on event placement language (from project brief):
  USCF crosstables are NOT shown in tiebreak/prize order.  We never
  claim "official 2nd place" — only "top crosstable finish" or
  "approximate top-5 by displayed crosstable order."  Wording fixes
  for that live in `scripts/20_score_underrated_potential.py` and
  `app.py`.
"""

import math
import os
import re
import sys

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


# ============================================================================
# Curated city table (~80 chess hubs).  Lat/Lon are city centroids.
# Coverage is intentionally biased toward cities that appear in our cached
# USCF data (Northeast + CA + MO + AZ + IL + TX + OH + GA + WA + OR + CO).
# For any city not listed here, we fall back to the state centroid.
# ============================================================================
CITY_COORDS: dict[tuple[str, str], tuple[float, float]] = {
    # New York / NJ / CT
    ("NEW YORK", "NY"):           (40.7128, -74.0060),
    ("BROOKLYN", "NY"):           (40.6782, -73.9442),
    ("QUEENS", "NY"):             (40.7282, -73.7949),
    ("BRONX", "NY"):              (40.8448, -73.8648),
    ("STATEN ISLAND", "NY"):      (40.5795, -74.1502),
    ("LONG ISLAND CITY", "NY"):   (40.7447, -73.9485),
    ("YONKERS", "NY"):            (40.9312, -73.8987),
    ("WHITE PLAINS", "NY"):       (41.0339, -73.7629),
    ("ALBANY", "NY"):             (42.6526, -73.7562),
    ("ROCHESTER", "NY"):          (43.1566, -77.6088),
    ("BUFFALO", "NY"):            (42.8864, -78.8784),
    ("PARSIPPANY", "NJ"):         (40.8579, -74.4260),
    ("NEWARK", "NJ"):             (40.7357, -74.1724),
    ("JERSEY CITY", "NJ"):        (40.7178, -74.0431),
    ("EDISON", "NJ"):             (40.5187, -74.4121),
    ("GLEN ROCK", "NJ"):          (40.9626, -74.1326),
    ("WEST ORANGE", "NJ"):        (40.7984, -74.2390),
    ("PRINCETON", "NJ"):          (40.3573, -74.6672),
    ("MORRISTOWN", "NJ"):         (40.7968, -74.4815),
    ("HARTFORD", "CT"):           (41.7637, -72.6851),
    ("STAMFORD", "CT"):           (41.0534, -73.5387),
    ("NEW HAVEN", "CT"):          (41.3083, -72.9279),
    # New England
    ("BOSTON", "MA"):             (42.3601, -71.0589),
    ("CAMBRIDGE", "MA"):          (42.3736, -71.1097),
    ("WALTHAM", "MA"):            (42.3765, -71.2356),
    ("WORCESTER", "MA"):          (42.2626, -71.8023),
    ("SOMERVILLE", "MA"):         (42.3876, -71.0995),
    ("MARLBOROUGH", "MA"):        (42.3459, -71.5523),
    ("PROVIDENCE", "RI"):         (41.8240, -71.4128),
    ("PORTLAND", "ME"):           (43.6591, -70.2568),
    ("MANCHESTER", "NH"):         (42.9956, -71.4548),
    ("BURLINGTON", "VT"):         (44.4759, -73.2121),
    # Mid-Atlantic
    ("PHILADELPHIA", "PA"):       (39.9526, -75.1652),
    ("PITTSBURGH", "PA"):         (40.4406, -79.9959),
    ("KING OF PRUSSIA", "PA"):    (40.0884, -75.3837),
    ("HARRISBURG", "PA"):         (40.2732, -76.8867),
    ("BALTIMORE", "MD"):          (39.2904, -76.6122),
    ("ROCKVILLE", "MD"):          (39.0840, -77.1528),
    ("BETHESDA", "MD"):           (38.9847, -77.0947),
    ("WASHINGTON", "DC"):         (38.9072, -77.0369),
    ("RICHMOND", "VA"):           (37.5407, -77.4360),
    ("ARLINGTON", "VA"):          (38.8816, -77.0910),
    ("VIRGINIA BEACH", "VA"):     (36.8529, -75.9780),
    ("WILMINGTON", "DE"):         (39.7391, -75.5398),
    # Southeast
    ("CHARLOTTE", "NC"):          (35.2271, -80.8431),
    ("RALEIGH", "NC"):            (35.7796, -78.6382),
    ("DURHAM", "NC"):             (35.9940, -78.8986),
    ("GREENSBORO", "NC"):         (36.0726, -79.7920),
    ("ASHEVILLE", "NC"):          (35.5951, -82.5515),
    ("ATLANTA", "GA"):            (33.7490, -84.3880),
    ("ALPHARETTA", "GA"):         (34.0754, -84.2941),
    ("SAVANNAH", "GA"):           (32.0809, -81.0912),
    ("ORLANDO", "FL"):            (28.5383, -81.3792),
    ("MIAMI", "FL"):              (25.7617, -80.1918),
    ("TAMPA", "FL"):              (27.9506, -82.4572),
    ("JACKSONVILLE", "FL"):       (30.3322, -81.6557),
    ("CHARLESTON", "SC"):         (32.7765, -79.9311),
    ("COLUMBIA", "SC"):           (34.0007, -81.0348),
    ("NASHVILLE", "TN"):          (36.1627, -86.7816),
    ("MEMPHIS", "TN"):            (35.1495, -90.0490),
    ("KNOXVILLE", "TN"):          (35.9606, -83.9207),
    ("LEXINGTON", "KY"):          (38.0406, -84.5037),
    ("LOUISVILLE", "KY"):         (38.2527, -85.7585),
    ("BIRMINGHAM", "AL"):         (33.5186, -86.8104),
    # Midwest / Heartland
    ("CHICAGO", "IL"):            (41.8781, -87.6298),
    ("NAPERVILLE", "IL"):         (41.7508, -88.1535),
    ("EVANSTON", "IL"):           (42.0451, -87.6877),
    ("DETROIT", "MI"):            (42.3314, -83.0458),
    ("ANN ARBOR", "MI"):          (42.2808, -83.7430),
    ("GRAND RAPIDS", "MI"):       (42.9634, -85.6681),
    ("COLUMBUS", "OH"):           (39.9612, -82.9988),
    ("CLEVELAND", "OH"):          (41.4993, -81.6944),
    ("CINCINNATI", "OH"):         (39.1031, -84.5120),
    ("INDIANAPOLIS", "IN"):       (39.7684, -86.1581),
    ("MILWAUKEE", "WI"):          (43.0389, -87.9065),
    ("MADISON", "WI"):            (43.0731, -89.4012),
    ("MINNEAPOLIS", "MN"):        (44.9778, -93.2650),
    ("SAINT PAUL", "MN"):         (44.9537, -93.0900),
    ("ST PAUL", "MN"):            (44.9537, -93.0900),
    ("SAINT LOUIS", "MO"):        (38.6270, -90.1994),
    ("ST LOUIS", "MO"):           (38.6270, -90.1994),
    ("KANSAS CITY", "MO"):        (39.0997, -94.5786),
    ("OMAHA", "NE"):              (41.2565, -95.9345),
    ("DES MOINES", "IA"):         (41.5868, -93.6250),
    # Texas / South Central
    ("HOUSTON", "TX"):            (29.7604, -95.3698),
    ("DALLAS", "TX"):             (32.7767, -96.7970),
    ("AUSTIN", "TX"):             (30.2672, -97.7431),
    ("SAN ANTONIO", "TX"):        (29.4241, -98.4936),
    ("PLANO", "TX"):              (33.0198, -96.6989),
    ("FORT WORTH", "TX"):         (32.7555, -97.3308),
    ("OKLAHOMA CITY", "OK"):      (35.4676, -97.5164),
    ("TULSA", "OK"):              (36.1539, -95.9928),
    ("NEW ORLEANS", "LA"):        (29.9511, -90.0715),
    ("LITTLE ROCK", "AR"):        (34.7465, -92.2896),
    # Mountain / Plains
    ("DENVER", "CO"):             (39.7392, -104.9903),
    ("BOULDER", "CO"):            (40.0150, -105.2705),
    ("COLORADO SPRINGS", "CO"):   (38.8339, -104.8214),
    ("ALBUQUERQUE", "NM"):        (35.0844, -106.6504),
    ("PHOENIX", "AZ"):            (33.4484, -112.0740),
    ("MESA", "AZ"):               (33.4152, -111.8315),
    ("SCOTTSDALE", "AZ"):         (33.4942, -111.9261),
    ("TUCSON", "AZ"):             (32.2226, -110.9747),
    ("SALT LAKE CITY", "UT"):     (40.7608, -111.8910),
    ("LAS VEGAS", "NV"):          (36.1699, -115.1398),
    ("RENO", "NV"):               (39.5296, -119.8138),
    # West Coast
    ("LOS ANGELES", "CA"):        (34.0522, -118.2437),
    ("SAN DIEGO", "CA"):          (32.7157, -117.1611),
    ("SAN FRANCISCO", "CA"):      (37.7749, -122.4194),
    ("OAKLAND", "CA"):            (37.8044, -122.2712),
    ("BERKELEY", "CA"):           (37.8716, -122.2727),
    ("SAN JOSE", "CA"):           (37.3382, -121.8863),
    ("PALO ALTO", "CA"):          (37.4419, -122.1430),
    ("MOUNTAIN VIEW", "CA"):      (37.3861, -122.0839),
    ("SACRAMENTO", "CA"):         (38.5816, -121.4944),
    ("FRESNO", "CA"):             (36.7378, -119.7871),
    ("LONG BEACH", "CA"):         (33.7701, -118.1937),
    ("IRVINE", "CA"):             (33.6846, -117.8265),
    ("SANTA CLARA", "CA"):        (37.3541, -121.9552),
    ("SEATTLE", "WA"):            (47.6062, -122.3321),
    ("REDMOND", "WA"):            (47.6740, -122.1215),
    ("BELLEVUE", "WA"):           (47.6101, -122.2015),
    ("TACOMA", "WA"):             (47.2529, -122.4443),
    ("PORTLAND", "OR"):           (45.5152, -122.6784),
    ("EUGENE", "OR"):             (44.0521, -123.0868),
    ("ANCHORAGE", "AK"):          (61.2181, -149.9003),
    ("HONOLULU", "HI"):           (21.3099, -157.8581),
}

# US state centroids — fallback when city isn't in our curated list.
US_STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "AL": (32.806671, -86.791130),  "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221), "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564), "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371),  "DE": (39.318523, -75.507141),
    "FL": (27.766279, -81.686783),  "GA": (33.040619, -83.643074),
    "HI": (21.094318, -157.498337), "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137),  "IN": (39.849426, -86.258278),
    "IA": (42.011539, -93.210526),  "KS": (38.526600, -96.726486),
    "KY": (37.668140, -84.670067),  "LA": (31.169546, -91.867805),
    "ME": (44.693947, -69.381927),  "MD": (39.063946, -76.802101),
    "MA": (42.230171, -71.530106),  "MI": (43.326618, -84.536095),
    "MN": (45.694454, -93.900192),  "MS": (32.741646, -89.678696),
    "MO": (38.456085, -92.288368),  "MT": (46.921925, -110.454353),
    "NE": (41.125370, -98.268082),  "NV": (38.313515, -117.055374),
    "NH": (43.452492, -71.563896),  "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482), "NY": (42.165726, -74.948051),
    "NC": (35.630066, -79.806419),  "ND": (47.528912, -99.784012),
    "OH": (40.388783, -82.764915),  "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938), "PA": (40.590752, -77.209755),
    "RI": (41.680893, -71.511780),  "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828),  "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461),  "UT": (40.150032, -111.862434),
    "VT": (44.045876, -72.710686),  "VA": (37.769337, -78.169968),
    "WA": (47.400902, -121.490494), "WV": (38.491226, -80.954456),
    "WI": (44.268543, -89.616508),  "WY": (42.755966, -107.302490),
    "DC": (38.897438, -77.026817),  "PR": (18.220833, -66.590149),
    "VI": (18.335765, -64.896335),
}


# ============================================================================
# 1. Event header parser
# ============================================================================
EVENT_LABELS = [
    "Event", "Location", "Event Date(s)", "Sponsoring Affiliate",
    "Organizer", "Chief TD", "Stats",
]


def _field(text: str, label: str) -> str | None:
    """Find a labeled value in the event-summary block."""
    pattern = re.escape(label) + r"\s*\n+\s*([^\n]+)"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def parse_event_header(html: str) -> dict:
    text = BeautifulSoup(html, "html.parser").get_text()
    text = text.replace("\xa0", " ")

    out = {
        "event_name_raw": None,
        "event_location_raw": None,
        "event_city": None,
        "event_state": None,
        "event_zip": None,
        "event_country": "US",
        "event_location_confidence": "missing",
        "event_date_raw": None,
        "event_start_date": None,
        "event_end_date_full": None,
        "sponsoring_affiliate": None,
        "chief_td": None,
        "num_sections": None,
        "num_players": None,
    }

    # --- Event line: "<NAME> (<event_id>)" ---
    ev = _field(text, "Event")
    if ev:
        out["event_name_raw"] = re.sub(r"\s*\(\d+\)\s*$", "", ev).strip()

    # --- Location: "CITY, ST ZIP" (US) or "CITY, COUNTRY3" (intl) ---
    loc = _field(text, "Location")
    if loc:
        out["event_location_raw"] = loc
        # US pattern: CITY, ST ZIP[+4]
        m = re.match(r"^(.+?)\s*,\s*([A-Z]{2})\s+(\d{5})(?:-\d{4})?\s*$", loc)
        if m:
            out["event_city"] = m.group(1).strip().upper()
            out["event_state"] = m.group(2)
            out["event_zip"] = m.group(3)
            out["event_country"] = "US"
            out["event_location_confidence"] = "city+state+zip"
        else:
            # City, ST  (no zip)
            m = re.match(r"^(.+?)\s*,\s*([A-Z]{2})\s*$", loc)
            if m:
                out["event_city"] = m.group(1).strip().upper()
                out["event_state"] = m.group(2)
                out["event_country"] = "US"
                out["event_location_confidence"] = "city+state"
            else:
                # International:  CITY, COUNTRY3
                m = re.match(r"^(.+?)\s*,\s*([A-Z]{3,})\s*$", loc)
                if m:
                    out["event_city"] = m.group(1).strip().upper()
                    out["event_country"] = m.group(2)
                    out["event_state"] = None
                    out["event_location_confidence"] = "international"
                else:
                    out["event_location_confidence"] = "unparsed"

    # --- Event Date(s): "YYYY-MM-DD thru YYYY-MM-DD" or single date ---
    d = _field(text, "Event Date(s)")
    if d:
        out["event_date_raw"] = d
        dates = re.findall(r"(\d{4}-\d{2}-\d{2})", d)
        if dates:
            out["event_start_date"] = dates[0]
            out["event_end_date_full"] = dates[-1]

    # --- Sponsoring Affiliate: strip the trailing (Annnnnnn) code ---
    aff = _field(text, "Sponsoring Affiliate")
    if aff:
        out["sponsoring_affiliate"] = re.sub(r"\s*\([A-Z0-9]+\)\s*$", "", aff).strip()

    # --- Chief TD ---
    td = _field(text, "Chief TD") or _field(text, "Chief  TD")
    if td:
        out["chief_td"] = re.sub(r"\s*\(\d+\)\s*$", "", td).strip()

    # --- Stats: "<N> Section(s), <M> Players" ---
    stats = _field(text, "Stats")
    if stats:
        m = re.search(r"(\d+)\s+Section\(s\)", stats)
        if m:
            out["num_sections"] = int(m.group(1))
        m = re.search(r"(\d+)\s+Players", stats)
        if m:
            out["num_players"] = int(m.group(1))

    return out


# ============================================================================
# 2. Offline geocoder
# ============================================================================
def geocode(city, state) -> tuple[float | None, float | None, str]:
    """Return (lat, lon, source) for a (city, state) pair.  Pandas NaN-safe.

    source = 'city_centroid' | 'state_centroid' | 'unresolved'
    """
    def _clean(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        s = str(x).strip()
        return s if s else None

    city, state = _clean(city), _clean(state)
    if city and state:
        coords = CITY_COORDS.get((city.upper(), state.upper()))
        if coords:
            return coords[0], coords[1], "city_centroid"
    if state:
        coords = US_STATE_CENTROIDS.get(state.upper())
        if coords:
            return coords[0], coords[1], "state_centroid"
    return None, None, "unresolved"


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two points, in statute miles."""
    R = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ============================================================================
# 3. Main pipeline
# ============================================================================
def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cache_dir = os.path.join(base, "data", "raw", "html_cache")
    proc_dir = os.path.join(base, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)

    raw_games_path = os.path.join(base, "data", "raw", "tables", "raw_games.csv")
    event_player_path = os.path.join(proc_dir, "event_player_scores.csv")
    profiles_path = os.path.join(proc_dir, "player_profiles.csv")

    print("=" * 60)
    print("PHASE 18: EVENT GEOGRAPHY & TRAVEL FEATURES")
    print("=" * 60)

    raw_games = pd.read_csv(raw_games_path, dtype=str)
    unique_events = raw_games[["event_id", "player_id"]].drop_duplicates()

    # --- 1. Parse event header for every unique event ---
    print(f"\n[1/4] Parsing event headers from {unique_events['event_id'].nunique()} cached crosstables ...")
    seen = set()
    rows = []
    for _, r in unique_events.iterrows():
        eid = str(r["event_id"])
        if eid in seen:
            continue
        cache_file = os.path.join(cache_dir, f"XtblMain.php_{eid}.0-{r['player_id']}.html")
        if not os.path.exists(cache_file):
            continue
        with open(cache_file, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        header = parse_event_header(html)
        header["event_id"] = eid
        rows.append(header)
        seen.add(eid)

    events_df = pd.DataFrame(rows)
    print(f"  parsed {len(events_df)} event headers")
    # Sanity: how many have location?
    have_loc = events_df["event_location_raw"].notna().sum()
    have_city_state = events_df["event_state"].notna().sum()
    print(f"  with raw Location field   : {have_loc} ({have_loc / len(events_df) * 100:.1f}%)")
    print(f"  parsed to city+state      : {have_city_state} ({have_city_state / len(events_df) * 100:.1f}%)")

    # --- 2. Geocode each event ---
    print("\n[2/4] Geocoding events (offline, curated city table + state centroid) ...")
    lats, lons, sources = [], [], []
    for _, ev in events_df.iterrows():
        lat, lon, src = geocode(ev.get("event_city"), ev.get("event_state"))
        lats.append(lat); lons.append(lon); sources.append(src)
    events_df["event_lat"] = lats
    events_df["event_lon"] = lons
    events_df["event_geo_source"] = sources

    by_src = events_df["event_geo_source"].value_counts()
    print("  geocoding source distribution:")
    for k, v in by_src.items():
        print(f"    {k:18s}: {v} events")

    event_meta_path = os.path.join(proc_dir, "event_metadata.csv")
    events_df.to_csv(event_meta_path, index=False)
    print(f"  wrote {event_meta_path}")

    # --- 3. Per-player travel features ---
    print("\n[3/4] Computing per-player travel features ...")
    # Merge game rows -> event geo
    games = raw_games[["player_id", "event_id"]].drop_duplicates()
    games["event_id"] = games["event_id"].astype(str)
    games["player_id"] = games["player_id"].astype(str)
    events_keep = events_df[["event_id", "event_state", "event_city",
                             "event_lat", "event_lon", "event_geo_source"]].copy()
    games = games.merge(events_keep, on="event_id", how="left")

    # Player's inferred home region — pull from existing player_profiles if available
    home_lookup = {}
    if os.path.exists(profiles_path):
        prof = pd.read_csv(profiles_path, dtype={"player_id": str})
        for _, r in prof.iterrows():
            home_lookup[str(r["player_id"])] = r.get("inferred_home_region")

    travel_rows = []
    for pid, g in games.groupby("player_id"):
        home_state = home_lookup.get(pid)
        home_lat, home_lon, home_src = geocode(None, home_state)

        # Unique events with a geocoded location
        g_geo = g.dropna(subset=["event_lat", "event_lon"])

        unique_event_locations = g_geo[["event_city", "event_state"]].drop_duplicates().shape[0]
        unique_event_states    = g_geo["event_state"].dropna().nunique()

        distances = []
        outside_home = 0
        total_events_with_state = 0
        if home_lat is not None:
            for _, row in g_geo.iterrows():
                d = haversine_miles(home_lat, home_lon, row["event_lat"], row["event_lon"])
                distances.append(d)
                if row["event_state"] and row["event_state"] != home_state:
                    outside_home += 1
                if row["event_state"]:
                    total_events_with_state += 1

        avg_travel = float(np.mean(distances)) if distances else None
        max_travel = float(np.max(distances))  if distances else None
        pct_outside = (outside_home / total_events_with_state) if total_events_with_state else None

        # Label
        label = "Insufficient location data"
        confidence = "low"
        if avg_travel is not None:
            if avg_travel < 30:
                label = "Local grinder"
            elif avg_travel < 100:
                label = "Regional traveler"
            elif avg_travel < 500:
                label = "Multi-state competitor"
            else:
                label = "Tournament road warrior"
            # confidence depends on whether we hit city centroids or just state
            city_share = (g_geo["event_geo_source"] == "city_centroid").mean() if len(g_geo) else 0
            confidence = "approximate (city-level)" if city_share >= 0.5 else "approximate (state-level)"

        travel_rows.append({
            "player_id": pid,
            "inferred_home_region": home_state,
            "unique_event_locations": unique_event_locations,
            "unique_event_states": unique_event_states,
            "avg_travel_distance_miles": avg_travel,
            "max_travel_distance_miles": max_travel,
            "pct_events_outside_home_region": pct_outside,
            "traveling_competitor_label": label,
            "travel_distance_confidence": confidence,
        })

    travel_df = pd.DataFrame(travel_rows)
    travel_path = os.path.join(proc_dir, "player_travel_features.csv")
    travel_df.to_csv(travel_path, index=False)
    print(f"  wrote {travel_path} ({len(travel_df)} players)")

    # --- 4. Merge travel features into player_profiles.csv ---
    print("\n[4/4] Merging travel features into player_profiles.csv ...")
    if not os.path.exists(profiles_path):
        print("  ⚠ player_profiles.csv missing — run scripts/19 first.  Skipping merge.")
        return
    prof = pd.read_csv(profiles_path, dtype={"player_id": str})

    # Drop old stub columns so the merge supersedes them cleanly
    new_cols = ["unique_event_locations", "avg_travel_distance_miles",
                "max_travel_distance_miles", "pct_events_outside_home_region",
                "traveling_competitor_label", "travel_distance_confidence",
                "unique_event_states"]
    drop_cols = [c for c in new_cols if c in prof.columns]
    prof = prof.drop(columns=drop_cols)

    merged = prof.merge(travel_df.drop(columns=["inferred_home_region"]),
                        on="player_id", how="left")
    merged.to_csv(profiles_path, index=False)
    print(f"  updated {profiles_path} with travel columns: {new_cols}")

    print("\n--- Sample (top 10 by max travel distance) ---")
    show = merged.dropna(subset=["max_travel_distance_miles"]).sort_values(
        "max_travel_distance_miles", ascending=False
    ).head(10)
    print(show[["player_id", "inferred_home_region", "unique_event_locations",
                "avg_travel_distance_miles", "max_travel_distance_miles",
                "traveling_competitor_label"]].to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
