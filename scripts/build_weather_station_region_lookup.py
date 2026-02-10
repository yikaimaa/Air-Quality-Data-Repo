"""
Nearest-neighbor mapping: weather station -> nearest PM2.5 station (and its region),
using lat/lon + haversine distance, with a distance threshold.

Inputs:
- pm25 station-region lookup (must include: region, latitude, longitude, and ideally station id)
- weather stations lat/lon file (must include: weather station id, latitude, longitude)

Output:
- weather_station_region_lookup_nn.csv

Recommended threshold for Ontario: 75 km (tune after inspecting distance stats).
"""

import pandas as pd
import numpy as np

# ---------------- paths (edit to match your repo structure if needed) ----------------
PM25_LOOKUP_PATH = "../Datasets/Ontario/pm25_station_region_lookup.csv"
WEATHER_LATLON_PATH = "../Datasets/Ontario/weather_stations_latlon.csv" 
OUT_PATH = "../Datasets/Ontario/weather_station_region_lookup_nn.csv"

DIST_THRESHOLD_KM = 75.0  # recommended starting point


# ---------------- helpers ----------------
def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """Pick the first column name in `candidates` that exists in df (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of {candidates} in columns: {list(df.columns)}")
    return None


def haversine_km_vec(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Vectorized haversine distance (km) from scalar (lat1, lon1) to arrays (lat2, lon2).
    """
    R = 6371.0
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# ---------------- load ----------------
pm25 = pd.read_csv(PM25_LOOKUP_PATH)
wlatlon = pd.read_csv(WEATHER_LATLON_PATH)

# ---------------- infer columns ----------------
# PM2.5 lookup must have: region, lat, lon; station id is optional but preferred
pm_region_col = pick_col(pm25, ["region", "Region", "REGION"])
pm_lat_col = pick_col(pm25, ["latitude", "lat", "Latitude", "LAT"])
pm_lon_col = pick_col(pm25, ["longitude", "lon", "Longitude", "LON", "long"])

pm_station_col = pick_col(
    pm25,
    ["pm25_station_id", "station_id", "station", "StationID", "STATION_ID", "id"],
    required=False
)

# Weather latlon must have: station id, lat, lon
w_station_col = pick_col(
    wlatlon,
    ["weather_station_id", "station_id", "station", "StationID", "STATION_ID", "id"]
)
w_lat_col = pick_col(wlatlon, ["latitude", "lat", "Latitude", "LAT"])
w_lon_col = pick_col(wlatlon, ["longitude", "lon", "Longitude", "LON", "long"])

# ---------------- clean coords ----------------
pm25[pm_lat_col] = pd.to_numeric(pm25[pm_lat_col], errors="coerce")
pm25[pm_lon_col] = pd.to_numeric(pm25[pm_lon_col], errors="coerce")
wlatlon[w_lat_col] = pd.to_numeric(wlatlon[w_lat_col], errors="coerce")
wlatlon[w_lon_col] = pd.to_numeric(wlatlon[w_lon_col], errors="coerce")

pm25_clean = pm25.dropna(subset=[pm_region_col, pm_lat_col, pm_lon_col]).copy()
wlatlon_clean = wlatlon.dropna(subset=[w_station_col, w_lat_col, w_lon_col]).copy()

if pm25_clean.empty:
    raise ValueError("PM2.5 lookup has no rows with valid (region, lat, lon).")
if wlatlon_clean.empty:
    raise ValueError("Weather latlon file has no rows with valid (station_id, lat, lon).")

# Ensure unique weather station rows
wlatlon_clean = wlatlon_clean.drop_duplicates(subset=[w_station_col]).copy()

# ---------------- prepare PM2.5 arrays for fast distance compute ----------------
pm_lat_arr = pm25_clean[pm_lat_col].to_numpy()
pm_lon_arr = pm25_clean[pm_lon_col].to_numpy()
pm_region_arr = pm25_clean[pm_region_col].astype(str).to_numpy()

if pm_station_col is not None:
    pm_station_arr = pm25_clean[pm_station_col].astype(str).to_numpy()
else:
    pm_station_arr = np.array([""] * len(pm25_clean), dtype=object)

# ---------------- nearest neighbor mapping ----------------
rows = []
dists_all = []

for _, r in wlatlon_clean.iterrows():
    w_id = str(r[w_station_col])
    w_lat = float(r[w_lat_col])
    w_lon = float(r[w_lon_col])

    dists = haversine_km_vec(w_lat, w_lon, pm_lat_arr, pm_lon_arr)
    j = int(np.argmin(dists))
    best_dist = float(dists[j])

    dists_all.append(best_dist)

    mapped_region = pm_region_arr[j] if best_dist <= DIST_THRESHOLD_KM else None
    mapped_pm_station = pm_station_arr[j] if (best_dist <= DIST_THRESHOLD_KM and pm_station_col is not None) else None

    rows.append({
        "weather_station_id": w_id,
        "weather_latitude": w_lat,
        "weather_longitude": w_lon,
        "nearest_pm25_station_id": mapped_pm_station,
        "assigned_region": mapped_region,
        "nn_distance_km": best_dist,
        "within_threshold": best_dist <= DIST_THRESHOLD_KM
    })

lookup = pd.DataFrame(rows)

# ---------------- save ----------------
lookup.to_csv(OUT_PATH, index=False)

# ---------------- diagnostics ----------------
print(f"Saved: {OUT_PATH}")
print(f"Weather stations processed: {len(lookup):,}")
print(f"Assigned (within {DIST_THRESHOLD_KM:.0f} km): {lookup['within_threshold'].sum():,}")
print(f"Unassigned (too far): {(~lookup['within_threshold']).sum():,}")

print("\nDistance (km) summary:")
print(pd.Series(dists_all).describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

print("\nTop 10 assigned regions by weather-station count:")
print(
    lookup.loc[lookup["within_threshold"] & lookup["assigned_region"].notna(), "assigned_region"]
    .value_counts()
    .head(10)
)

print("\nFarthest 10 weather stations (by NN distance):")
print(lookup.sort_values("nn_distance_km", ascending=False).head(10)[
    ["weather_station_id", "assigned_region", "nn_distance_km", "within_threshold"]
])
