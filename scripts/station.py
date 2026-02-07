import requests
import time
import os

BASE_URL = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"

STATION_ID_FILE = "on_station_ids.txt"  
START_YEAR = 2020
END_YEAR = 2025

# 读取 stationID 列表
with open(STATION_ID_FILE, "r") as f:
    station_ids = [line.strip() for line in f if line.strip()]

print(f"Total stations: {len(station_ids)}")

params_base = {
    "format": "csv",
    "Year": None,
    "Month": "1",
    "Day": "1",
    "timeframe": "2",  # daily
    "submit": "Download Data"
}

for station_id in station_ids:
    print(f"\n=== Processing station {station_id} ===")

    # 每个 station 一个文件夹
    output_dir = f"station_{station_id}"
    os.makedirs(output_dir, exist_ok=True)

    for year in range(START_YEAR, END_YEAR + 1):
        params = params_base.copy()
        params["stationID"] = station_id
        params["Year"] = str(year)

        print(f"Downloading {station_id} {year}...")

        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"Failed {station_id} {year}: {e}")
            continue

        filename = f"station_{station_id}_daily_{year}.csv"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "wb") as f:
            f.write(r.content)

        time.sleep(1)  # 防止限流
