import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ---- CONFIG ----
start = "20200101"     # YYYYMMDD
end   = "20241231"

# District coordinates
districts = {
    "Bengaluru": {"lat": 12.97, "lon": 77.59},
    "Mysuru":    {"lat": 12.30, "lon": 76.64},
    "Raichur":   {"lat": 16.20, "lon": 77.35},
}

# NASA POWER URL
BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# NASA parameters
PARAMS = ",".join([
    "T2M",                # temperature
    "RH2M",               # humidity
    "WS2M",               # wind speed
    "PS",                 # surface pressure
    "ALLSKY_SFC_SW_DWN"   # solar radiation
])

all_dfs = []

def compute_air_density(temp_C, pressure_mb, humidity_pct):
    """Compute air density using ideal gas law & humidity correction"""
    T = temp_C + 273.15  # K
    P = pressure_mb * 100  # Pa
    RH = humidity_pct / 100
    R_dry = 287.05
    R_v = 461.495
    Pv = RH * 0.611 * np.exp(17.27 * temp_C / (temp_C + 237.3)) * 1000
    Pd = P - Pv
    return (Pd / (R_dry * T)) + (Pv / (R_v * T))


def compute_wind_power(wind_speed, air_density):
    """Wind turbine theoretical formula"""
    swept_area = np.pi * (50 ** 2)      # 50m radius turbine
    Cp = 0.40                            # power coefficient
    return 0.5 * air_density * swept_area * (wind_speed ** 3) * Cp / 1000  # kW


for name, loc in districts.items():
    lat = loc["lat"]
    lon = loc["lon"]

    print(f"\nFetching NASA data for {name}...")

    url = (
        f"{BASE_URL}"
        f"?start={start}&end={end}"
        f"&latitude={lat}&longitude={lon}"
        f"&parameters={PARAMS}"
        f"&community=RE"
        f"&format=JSON"
    )

    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    records = data["properties"]["parameter"]
    timestamps = list(records["T2M"].keys())

    df = pd.DataFrame({"datetime": timestamps})

    # Add NASA parameters
    for param, series in records.items():
        df[param] = df["datetime"].map(series)

    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H")
    df["district"] = name
    df["latitude"] = lat
    df["longitude"] = lon

    # Friendly names
    df = df.rename(columns={
        "T2M": "temperature_C",
        "RH2M": "humidity_pct",
        "WS2M": "wind_speed_ms",
        "PS": "pressure_mb",
        "ALLSKY_SFC_SW_DWN": "solar_radiation_Wm2",
    })

    # --- COMPUTE EXTRA COLUMNS ---
    df["air_density"] = compute_air_density(
        df["temperature_C"], df["pressure_mb"], df["humidity_pct"]
    )

    df["wind_power_kw"] = compute_wind_power(
        df["wind_speed_ms"], df["air_density"]
    )

    all_dfs.append(df)


# Combine all districts
full_df = pd.concat(all_dfs, ignore_index=True)
full_df = full_df.sort_values(["district", "datetime"])

# Select final 10 columns
final_columns = [
    "datetime", "district", "latitude", "longitude",
    "temperature_C", "pressure_mb", "humidity_pct",
    "wind_speed_ms", "solar_radiation_Wm2", "wind_power_kw"
]

full_df = full_df[final_columns]

# Save
output_file = "karnataka_real_dataset_2020_2024.csv"
full_df.to_csv(output_file, index=False)

print("\n====================================")
print("✔ Saved:", output_file)
print("✔ Total rows:", len(full_df))
print("✔ Preview:")
print(full_df.head())
print("====================================\n")
