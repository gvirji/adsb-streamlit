"""ADS-B Trajectory Analytics — Streamlit აპი
გაშვება:  streamlit run app.py
"""
import json, os, tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import streamlit as st

import prepare_data as pdata   # იგივე დამუშავების ლოგიკა, რაც ნოუთბუქში

st.set_page_config(page_title="ADS-B Trajectory Analytics", page_icon="✈️", layout="wide")

PHASE_COLORS = {"Climb": "#00A86B", "Level / Cruise": "#1E88E5", "Descent": "#E53935", "Unknown": "#9E9E9E"}
DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

# ---------------------------------------------------------------- მონაცემები
@st.cache_data(show_spinner="CSV მუშავდება…")
def process_csv(path: str) -> pd.DataFrame:
    return pdata.anomalies(pdata.dynamics(pdata.load(path)))

@st.cache_data
def load_json(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f)["points"])
    df = df.rename(columns={"fid": "flight_id", "t": "timestamp", "lat": "latitude", "lon": "longitude",
                            "alt": "altitude_ft", "gs": "ground_speed_kt", "hdg": "heading_deg",
                            "vr": "vertical_rate_fpm", "anom": "is_anomaly"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

with st.sidebar:
    st.title("✈️ ADS-B Analytics")
    up = st.file_uploader("ატვირთე adsb_hexagon_data.csv", type="csv")
    path = st.text_input("ან CSV-ის ლოკალური გზა", value="")
    if up is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(up.getvalue())
        df = process_csv(tmp.name)
    elif path and os.path.exists(path):
        df = process_csv(path)
    elif os.path.exists("data.json"):
        df = load_json("data.json")
        st.info("ნაჩვენებია დემო მონაცემები (data.json). ატვირთე CSV სრული ანალიზისთვის.")
    else:
        st.error("მონაცემები ვერ მოიძებნა. ატვირთე CSV."); st.stop()

    counts = df["icao"].value_counts()
    icao = st.selectbox("თვითმფრინავი (ICAO)", counts.index, format_func=lambda i: f"{i} ({counts[i]} წერტილი)")
    fl = df[df["icao"] == icao].groupby("flight_id").agg(n=("timestamp", "size"), start=("timestamp", "min"))
    fid = st.selectbox("ფრენა", fl.sort_values("n", ascending=False).index,
                       format_func=lambda i: f"{fl.loc[i, 'start']:%Y-%m-%d %H:%M} · {fl.loc[i, 'n']} წერტ.")

seg = df[(df["icao"] == icao) & (df["flight_id"] == fid)].sort_values("timestamp").reset_index(drop=True)

# ---------------------------------------------------------------- KPI
dur = (seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0]).total_seconds() / 60
c = st.columns(6)
c[0].metric("წერტილი", len(seg))
c[1].metric("ხანგრძლივობა", f"{dur:.1f} წთ")
c[2].metric("მაქს. სიმაღლე", f"{seg['altitude_ft'].max():,.0f} ft")
c[3].metric("მინ. სიმაღლე", f"{seg['altitude_ft'].min():,.0f} ft")
c[4].metric("საშ. სიჩქარე", "—" if seg["ground_speed_kt"].isna().all() else f"{seg['ground_speed_kt'].mean():.0f} kt")
c[5].metric("ანომალია", int(seg["is_anomaly"].sum()))

def hover(d):
    return [f"<b>Time:</b> {t}<br><b>Altitude:</b> {a:,.0f} ft<br><b>Speed:</b> {s:.1f} kt<br>"
            f"<b>Track:</b> {h:.1f}°<br><b>V/S:</b> {v:.0f} ft/min"
            for t, a, s, h, v in zip(d["timestamp"], d["altitude_ft"], d["ground_speed_kt"].fillna(0),
                                     d["heading_deg"].fillna(0), d["vertical_rate_fpm"].fillna(0))]

tabs = st.tabs(["🗺️ 2D რუკა", "🛰️ FlightRadar სტილი", "🧊 3D", "📈 პროფილები", "🚨 ანომალიები"])

# ---------------------------------------------------------------- 2D
with tabs[0]:
    an = seg[seg["is_anomaly"]]
    fig = go.Figure()
    fig.add_trace(go.Scattermap(lat=seg["latitude"], lon=seg["longitude"], mode="lines",
                                line=dict(width=4, color="#00A6FF"), hoverinfo="skip", name="Trajectory"))
    fig.add_trace(go.Scattermap(lat=seg["latitude"], lon=seg["longitude"], mode="markers", name="ADS-B positions",
                                marker=dict(size=11, color=seg["altitude_ft"], colorscale="Turbo", showscale=True,
                                            colorbar=dict(title="Altitude (ft)", thickness=14)),
                                text=hover(seg), hovertemplate="%{text}<extra></extra>"))
    if len(an):
        fig.add_trace(go.Scattermap(lat=an["latitude"], lon=an["longitude"], mode="markers", name="Anomaly",
                                    marker=dict(size=18, color="#FF1744"), text=hover(an), hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(map=dict(style="carto-darkmatter", zoom=6,
                               center=dict(lat=float(seg["latitude"].mean()), lon=float(seg["longitude"].mean()))),
                      height=700, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#0e1117", font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- PyDeck
with tabs[1]:
    r = seg.copy()
    lo, hi = r["altitude_ft"].min(), r["altitude_ft"].max()
    k = (r["altitude_ft"] - lo) / max(hi - lo, 1)
    r["r"], r["g"], r["b"] = (20 + 235 * k).astype(int), (230 - 70 * k).astype(int), (255 - 220 * k).astype(int)
    r["time"] = r["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for col, name in [("ground_speed_kt", "spd"), ("heading_deg", "hdg"), ("vertical_rate_fpm", "vr")]:
        r[name] = r[col].round(1).fillna(0).astype(str)
    route = pd.DataFrame({"path": [r[["longitude", "latitude"]].values.tolist()]})
    last = r.tail(1).assign(plane="✈", label=lambda d: f" {icao} | " + d["altitude_ft"].round(0).astype(int).astype(str) + " ft")
    layers = [
        pdk.Layer("PathLayer", route, get_path="path", get_color=[0, 220, 255, 60], get_width=12, width_min_pixels=9),
        pdk.Layer("PathLayer", route, get_path="path", get_color=[0, 230, 255, 235], get_width=4, width_min_pixels=3),
        pdk.Layer("ScatterplotLayer", r, get_position="[longitude, latitude]", get_radius=2200, radius_min_pixels=4,
                  radius_max_pixels=8, get_fill_color="[r, g, b, 230]", stroked=True, get_line_color=[255, 255, 255, 160],
                  line_width_min_pixels=1, pickable=True),
        pdk.Layer("TextLayer", last, get_position="[longitude, latitude]", get_text="plane", get_size=42,
                  get_color=[255, 215, 0, 255], get_angle="heading_deg", billboard=True),
        pdk.Layer("TextLayer", last, get_position="[longitude, latitude]", get_text="label", get_size=15,
                  get_color=[255, 255, 255, 255], get_pixel_offset=[45, -25], get_text_anchor="'start'"),
    ]
    tooltip = {"html": "<b>ADS-B POSITION</b><br><b>Time:</b> {time}<br><b>Altitude:</b> {altitude_ft} ft<br>"
                       "<b>Speed:</b> {spd} kt<br><b>Track:</b> {hdg}°<br><b>V/S:</b> {vr} ft/min",
               "style": {"backgroundColor": "#111827", "color": "white"}}
    st.pydeck_chart(pdk.Deck(layers=layers, map_style=DARK_STYLE, tooltip=tooltip,
                             initial_view_state=pdk.ViewState(latitude=float(r["latitude"].mean()),
                                                              longitude=float(r["longitude"].mean()), zoom=6, pitch=35)),
                    height=700)

# ---------------------------------------------------------------- 3D
with tabs[2]:
    z_scale = st.slider("სიმაღლის გაზვიადება", 0.5, 3.0, 1.5, 0.1)
    nrm, an = seg[~seg["is_anomaly"]], seg[seg["is_anomaly"]]
    f3 = go.Figure()
    f3.add_trace(go.Scatter3d(x=seg["longitude"], y=seg["latitude"], z=seg["altitude_ft"], mode="lines",
                              line=dict(width=6, color="#00A6FF"), hoverinfo="skip", name="Trajectory"))
    f3.add_trace(go.Scatter3d(x=nrm["longitude"], y=nrm["latitude"], z=nrm["altitude_ft"], mode="markers", name="Normal",
                              marker=dict(size=6, color="#00C853"), text=hover(nrm), hovertemplate="%{text}<extra></extra>"))
    if len(an):
        f3.add_trace(go.Scatter3d(x=an["longitude"], y=an["latitude"], z=an["altitude_ft"], mode="markers", name="Anomaly",
                                  marker=dict(size=11, color="#FF1744", symbol="diamond", line=dict(color="white", width=2)),
                                  text=hover(an), hovertemplate="%{text}<extra></extra>"))
    f3.update_layout(template="plotly_dark", height=720, margin=dict(l=0, r=0, t=10, b=0),
                     scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Altitude (ft)",
                                aspectratio=dict(x=1, y=1, z=z_scale)))
    st.plotly_chart(f3, use_container_width=True)

# ---------------------------------------------------------------- პროფილები
with tabs[3]:
    fp = make_subplots(rows=2, cols=2, vertical_spacing=0.16,
                       subplot_titles=("Altitude", "Ground speed", "Track", "Vertical rate"))
    for (col, ttl), (rr, cc) in zip([("altitude_ft", "ft"), ("ground_speed_kt", "kt"),
                                     ("heading_deg", "°"), ("vertical_rate_fpm", "ft/min")],
                                    [(1, 1), (1, 2), (2, 1), (2, 2)]):
        fp.add_trace(go.Scatter(x=seg["timestamp"], y=seg[col], mode="lines+markers", line=dict(width=3),
                                marker=dict(size=9, color=[PHASE_COLORS[p] for p in seg["phase"]])), row=rr, col=cc)
        fp.update_yaxes(title_text=ttl, row=rr, col=cc)
    fp.update_layout(height=700, showlegend=False, template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fp, use_container_width=True)
    st.caption("მარკერების ფერი = ფრენის ფაზა: 🟢 Climb · 🔵 Level/Cruise · 🔴 Descent · ⚪ Unknown")
    st.dataframe(seg[["timestamp", "altitude_ft", "ground_speed_kt", "heading_deg", "vertical_rate_fpm", "phase"]],
                 use_container_width=True)

# ---------------------------------------------------------------- ყველა ანომალია
with tabs[4]:
    a = df[df["is_anomaly"]]
    st.write(f"სულ **{len(a)}** ანომალიური წერტილი {len(df):,}-დან.")
    fa = go.Figure()
    n_ = df[~df["is_anomaly"]]
    fa.add_trace(go.Scattermap(lat=n_["latitude"], lon=n_["longitude"], mode="markers", name="Normal",
                               marker=dict(size=5, color="#00C853", opacity=0.4), hoverinfo="skip"))
    fa.add_trace(go.Scattermap(lat=a["latitude"], lon=a["longitude"], mode="markers", name="Anomaly",
                               marker=dict(size=14, color="#FF1744"), text=hover(a), hovertemplate="%{text}<extra></extra>"))
    fa.update_layout(map=dict(style="carto-darkmatter", zoom=5,
                              center=dict(lat=float(df["latitude"].mean()), lon=float(df["longitude"].mean()))),
                     height=650, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#0e1117", font=dict(color="white"))
    st.plotly_chart(fa, use_container_width=True)
    st.dataframe(a[["icao", "flight_id", "timestamp", "altitude_ft", "ground_speed_kt", "vertical_rate_fpm"]],
                 use_container_width=True)
    st.download_button("⬇️ დამუშავებული მონაცემები (CSV)", df.to_csv(index=False).encode(), "adsb_processed.csv", "text/csv")
