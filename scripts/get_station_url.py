import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

BASE_URL = "https://climate.weather.gc.ca/historical_data/search_historic_data_stations_e.html"

def fetch_station_ids_on_all_pages(
    prov="ON",
    timeframe=1,            # 1=Hourly
    start_year=2020,
    end_year=2026,
    year=2026,
    month=1,
    day=30,
    rows_per_page=25,
    sleep_sec=0.3,
):
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; StationID-Scraper/1.0)"
    })

    station_ids = set()
    start_row = 1  

    while True:
        params = {
            "searchType": "stnProv",
            "timeframe": str(timeframe),
            "lstProvince": prov,
            "optLimit": "yearRange",
            "StartYear": str(start_year),
            "EndYear": str(end_year),
            "Year": str(year),
            "Month": str(month),
            "Day": str(day),
            "selRowPerPage": str(rows_per_page),
            "txtCentralLatMin": "0",
            "txtCentralLatSec": "0",
            "txtCentralLongMin": "0",
            "txtCentralLongSec": "0",
            "startRow": str(start_row),
        }

        resp = session.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        found = set()

   
        for a in soup.select('a[href*="StationID="]'):
            href = a.get("href", "")
            qs = parse_qs(urlparse(href).query)
            if "StationID" in qs and qs["StationID"]:
                found.add(qs["StationID"][0])

        for inp in soup.select('input[name="StationID"], input[name="stationID"], input[name="stationId"]'):
            val = (inp.get("value") or "").strip()
            if val:
                found.add(val)


        found.update(re.findall(r"StationID=(\d+)", html))
        found = {x for x in found if str(x).isdigit()}
        if len(found) == 0:
            break

        before = len(station_ids)
        station_ids.update(found)
        after = len(station_ids)
        if after == before:
            break

        start_row += rows_per_page
        time.sleep(sleep_sec)

    return sorted(station_ids, key=int)


if __name__ == "__main__":
    ids = fetch_station_ids_on_all_pages(prov="ON")
    print("Total StationIDs:", len(ids))
    print("First 20:", ids[:20])
    with open("ON_station_ids.txt", "w", encoding="utf-8") as f:
        for sid in ids:
            f.write(sid + "\n")
    print("Saved to ON_station_ids.txt")
