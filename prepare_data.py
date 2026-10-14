"""CSV (adsb_hexagon_data.csv) -> data.json  (იგივე ლოგიკა, რაც ნოუთბუქში)
გამოყენება:  python prepare_data.py adsb_hexagon_data.csv
"""
import csv, sys, json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

GAP_MIN = 20          # ახალი ფრენა, თუ ორ პოზიციას შორის > 20 წთ
CONTAMINATION = 0.05  # Isolation Forest-ის ანომალიების წილი
R_KM = 6371.0088

def load(path):
    csv.field_size_limit(2**31 - 1)  # iq/magnitudes/phases სვეტები დიდია
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            if len(r) < 11:
                continue
            rows.append((r[2], r[6], r[8], r[9], r[10]))
    df = pd.DataFrame(rows, columns=["icao", "timestamp", "latitude", "longitude", "altitude_ft"])
    for c in ["latitude", "longitude", "altitude_ft"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["icao"] = df["icao"].astype(str).str.strip().str.upper()
    return (df.dropna(subset=["timestamp", "latitude", "longitude", "altitude_ft"])
              .sort_values(["icao", "timestamp"]).reset_index(drop=True))

def dynamics(df):
    g = df.groupby("icao")
    gap = g["timestamp"].diff().dt.total_seconds() / 60
    df["flight_id"] = (gap.fillna(0) > GAP_MIN).groupby(df["icao"]).cumsum().astype(int)
    key = [df["icao"], df["flight_id"]]
    g = df.groupby(["icao", "flight_id"])
    dt = g["timestamp"].diff().dt.total_seconds()
    lat1 = np.radians(g["latitude"].shift(1).to_numpy(float)); lon1 = np.radians(g["longitude"].shift(1).to_numpy(float))
    lat2 = np.radians(df["latitude"].to_numpy(float));        lon2 = np.radians(df["longitude"].to_numpy(float))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * R_KM * np.arcsin(np.sqrt(a))
    df["ground_speed_kt"] = dist / (dt / 3600) / 1.852
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    df["heading_deg"] = (np.degrees(np.arctan2(y, x)) + 360) % 360
    df["vertical_rate_fpm"] = g["altitude_ft"].diff() / (dt / 60)
    df["phase"] = np.select([df["vertical_rate_fpm"].isna(), df["vertical_rate_fpm"] > 300, df["vertical_rate_fpm"] < -300],
                            ["Unknown", "Climb", "Descent"], "Level / Cruise")
    return df

def anomalies(df):
    feats = ["altitude_ft", "ground_speed_kt", "vertical_rate_fpm"]
    ok = df.dropna(subset=feats)
    df["is_anomaly"] = False
    if len(ok) >= 5:
        X = RobustScaler().fit_transform(ok[feats])
        iso = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=42)
        df.loc[ok.index, "is_anomaly"] = iso.fit_predict(X) == -1
    return df

def r(v, n):
    return None if pd.isna(v) else round(float(v), n)

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "adsb_hexagon_data.csv"
    df = anomalies(dynamics(load(src)))
    pts = [dict(icao=p.icao, fid=int(p.flight_id), t=p.timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                lat=r(p.latitude, 5), lon=r(p.longitude, 5), alt=int(p.altitude_ft),
                gs=r(p.ground_speed_kt, 1), hdg=r(p.heading_deg, 1), vr=r(p.vertical_rate_fpm, 0),
                phase=p.phase, anom=bool(p.is_anomaly))
           for p in df.itertuples()]
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump({"points": pts}, f, separators=(",", ":"))
    print(f"✅ data.json: {len(pts)} წერტილი, {df['icao'].nunique()} თვითმფრინავი, {int(df['is_anomaly'].sum())} ანომალია")
