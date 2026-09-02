"""
RailPulse -- Dynamic ETA forecasting for Indian Railways coaching trains
SIH 2026 problem statement SIH26028 (Ministry of Railways)

Run with:
    streamlit run app.py

Uses simulated data -- real GPS/NTES feeds are not publicly available,
so this demonstrates the approach rather than live production data.
"""

import random
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RailPulse: Dynamic ETA for Coaching Trains", page_icon="🚆", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


def to_clock_and_day(start_clock, minutes_from_start):
    sh, sm = (int(x) for x in start_clock.split(":"))
    total = sh * 60 + sm + round(minutes_from_start)
    day_offset = total // 1440
    minute_of_day = total % 1440
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}", day_offset


def status_of(delay):
    if delay > 30:
        return "danger", "Major delay", "🔴"
    if delay > 10:
        return "caution", "Delayed", "🟡"
    if delay < -2:
        return "clear", "Running early", "🟢"
    return "clear", "On time", "🟢"


def normalize(weights):
    total = sum(weights.values())
    return {k: round(v / total * 1000) / 10 for k, v in weights.items()}


def initial_factors():
    raw = {
        "historical": 30 + random.random() * 20,
        "congestion": 15 + random.random() * 20,
        "weather": 5 + random.random() * 15,
        "preceding": 10 + random.random() * 15,
        "speed_restriction": 5 + random.random() * 10,
    }
    return normalize(raw)


def reroll_factors(factors):
    raw = {k: max(1.0, v + (random.random() - 0.5) * 6) for k, v in factors.items()}
    return normalize(raw)


def confidence_minutes(stations_ahead, train):
    base, per_station = 2, 1.4
    weather_factor = (train["factors"]["weather"] / 100) * 8
    return round(base + stations_ahead * per_station + weather_factor)


FACTOR_LABELS = {
    "historical": "Historical section running time",
    "congestion": "Live downstream congestion",
    "weather": "Weather conditions",
    "preceding": "Preceding train delay propagation",
    "speed_restriction": "Temporary speed restrictions",
}

PALETTE = {
    "bg": "#0A0F14",
    "surface": "#111922",
    "surface2": "#16202B",
    "border": "#24313D",
    "text": "#E8EDF1",
    "muted": "#8996A3",
    "accent": "#4FD1C5",
    "clear": "#3FB878",
    "caution": "#E8B93D",
    "danger": "#E15252",
}


# ---------------------------------------------------------------------------
# Train data
# ---------------------------------------------------------------------------

def base_train_data():
    return [
        {
            "no": "12951", "name": "Mumbai Rajdhani Express", "zone": "Western Railway",
            "from": "Mumbai Central", "to": "New Delhi", "start_clock": "17:00", "speed": 112,
            "stations": [
                {"code": "BCT", "name": "Mumbai Central", "km": 0, "sched": 0},
                {"code": "BRC", "name": "Vadodara", "km": 392, "sched": 330},
                {"code": "RTM", "name": "Ratlam", "km": 549, "sched": 468},
                {"code": "KOTA", "name": "Kota", "km": 713, "sched": 612},
                {"code": "NDLS", "name": "New Delhi", "km": 1384, "sched": 985},
            ],
            "at_station": 2, "delay": 8.0,
        },
        {
            "no": "12841", "name": "Coromandel Express", "zone": "South Eastern Railway",
            "from": "Howrah", "to": "Chennai Central", "start_clock": "14:50", "speed": 88,
            "stations": [
                {"code": "HWH", "name": "Howrah", "km": 0, "sched": 0},
                {"code": "KGP", "name": "Kharagpur", "km": 115, "sched": 105},
                {"code": "BBS", "name": "Bhubaneswar", "km": 441, "sched": 375},
                {"code": "VSKP", "name": "Visakhapatnam", "km": 867, "sched": 690},
                {"code": "BZA", "name": "Vijayawada", "km": 1103, "sched": 900},
                {"code": "MAS", "name": "Chennai Central", "km": 1662, "sched": 1330},
            ],
            "at_station": 3, "delay": 22.0,
        },
        {
            "no": "12622", "name": "Tamil Nadu Express", "zone": "Northern Railway",
            "from": "New Delhi", "to": "Chennai Central", "start_clock": "22:30", "speed": 79,
            "stations": [
                {"code": "NDLS", "name": "New Delhi", "km": 0, "sched": 0},
                {"code": "AGC", "name": "Agra", "km": 200, "sched": 165},
                {"code": "BPL", "name": "Bhopal", "km": 700, "sched": 555},
                {"code": "NGP", "name": "Nagpur", "km": 1093, "sched": 870},
                {"code": "BZA", "name": "Vijayawada", "km": 1660, "sched": 1320},
                {"code": "MAS", "name": "Chennai Central", "km": 2180, "sched": 1730},
            ],
            "at_station": 1, "delay": -2.0,
        },
        {
            "no": "12809", "name": "Mumbai Howrah Mail", "zone": "Central Railway",
            "from": "Mumbai CST", "to": "Howrah", "start_clock": "23:45", "speed": 64,
            "stations": [
                {"code": "CSTM", "name": "Mumbai CST", "km": 0, "sched": 0},
                {"code": "BSL", "name": "Bhusaval", "km": 430, "sched": 375},
                {"code": "NGP", "name": "Nagpur", "km": 831, "sched": 690},
                {"code": "DURG", "name": "Durg", "km": 1099, "sched": 900},
                {"code": "R", "name": "Raipur", "km": 1131, "sched": 930},
                {"code": "BSP", "name": "Bilaspur", "km": 1229, "sched": 1005},
                {"code": "HWH", "name": "Howrah", "km": 1968, "sched": 1610},
            ],
            "at_station": 4, "delay": 34.0,
        },
        {
            "no": "12301", "name": "Howrah Rajdhani Express", "zone": "Eastern Railway",
            "from": "Howrah", "to": "New Delhi", "start_clock": "16:50", "speed": 118,
            "stations": [
                {"code": "HWH", "name": "Howrah", "km": 0, "sched": 0},
                {"code": "ASN", "name": "Asansol", "km": 214, "sched": 180},
                {"code": "GAYA", "name": "Gaya", "km": 430, "sched": 360},
                {"code": "DDU", "name": "Mughalsarai", "km": 587, "sched": 480},
                {"code": "CNB", "name": "Kanpur", "km": 987, "sched": 780},
                {"code": "NDLS", "name": "New Delhi", "km": 1447, "sched": 1085},
            ],
            "at_station": 0, "delay": 3.0,
        },
        {
            "no": "12723", "name": "Telangana Express", "zone": "South Central Railway",
            "from": "New Delhi", "to": "Hyderabad", "start_clock": "06:40", "speed": 71,
            "stations": [
                {"code": "NDLS", "name": "New Delhi", "km": 0, "sched": 0},
                {"code": "BPL", "name": "Bhopal", "km": 700, "sched": 570},
                {"code": "NGP", "name": "Nagpur", "km": 1093, "sched": 880},
                {"code": "SC", "name": "Secunderabad", "km": 1493, "sched": 1210},
                {"code": "HYB", "name": "Hyderabad", "km": 1503, "sched": 1225},
            ],
            "at_station": 3, "delay": 5.0,
        },
    ]


if "trains" not in st.session_state:
    _trains_init = base_train_data()
    for _t in _trains_init:
        _t["factors"] = initial_factors()
    st.session_state.trains = _trains_init
if "last_tick" not in st.session_state:
    st.session_state.last_tick = datetime.now()

trains = st.session_state.trains


def tick():
    for t in trains:
        drift = (random.random() - 0.42) * 3
        t["delay"] = clamp(t["delay"] + drift, -5, 75)
        t["factors"] = reroll_factors(t["factors"])
    st.session_state.last_tick = datetime.now()


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_route_figure(train):
    n = len(train["stations"])
    xs = list(range(n))
    codes = [s["code"] for s in train["stations"]]
    names = [s["name"] for s in train["stations"]]
    colors = [PALETTE["accent"] if i <= train["at_station"] else PALETTE["surface"] for i in range(n)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, n - 1], y=[0, 0], mode="lines",
                              line=dict(color=PALETTE["border"], width=3),
                              hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[0, train["at_station"] + 0.5], y=[0, 0], mode="lines",
                              line=dict(color=PALETTE["accent"], width=3),
                              hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=[0] * n, mode="markers+text",
                              marker=dict(size=16, color=colors, line=dict(color=PALETTE["accent"], width=2)),
                              text=codes, textposition="bottom center",
                              textfont=dict(color=PALETTE["muted"], size=11),
                              hovertext=names, hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=[train["at_station"] + 0.5], y=[0], mode="markers",
                              marker=dict(size=14, color=PALETTE["bg"], line=dict(color=PALETTE["accent"], width=3)),
                              showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        height=160, margin=dict(l=20, r=20, t=10, b=30),
        plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        xaxis=dict(visible=False, range=[-0.5, n - 0.5]),
        yaxis=dict(visible=False, range=[-1, 1]),
        font=dict(color=PALETTE["text"]),
    )
    return fig


def build_architecture_figure():
    boxes = [
        {"cx": 1.0, "w": 1.9, "label": "<b>Data sources</b><br>GPS, signalling,<br>weather, history"},
        {"cx": 3.4, "w": 1.7, "label": "<b>Feature engineering</b>"},
        {"cx": 5.9, "w": 2.1, "label": "<b>Prediction engine</b><br>Base model +<br>real-time correction"},
        {"cx": 8.3, "w": 1.6, "label": "<b>Serving APIs</b>"},
        {"cx": 10.5, "w": 1.9, "label": "<b>Consumers</b><br>Apps, displays,<br>control rooms"},
    ]
    fig = go.Figure()
    y0, y1 = -0.5, 0.5
    for b in boxes:
        x0, x1 = b["cx"] - b["w"] / 2, b["cx"] + b["w"] / 2
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      line=dict(color=PALETTE["border"], width=1.5), fillcolor=PALETTE["surface2"])
        fig.add_annotation(x=b["cx"], y=0, text=b["label"], showarrow=False,
                            font=dict(color=PALETTE["text"], size=12), align="center")
    for i in range(len(boxes) - 1):
        x_start = boxes[i]["cx"] + boxes[i]["w"] / 2
        x_end = boxes[i + 1]["cx"] - boxes[i + 1]["w"] / 2
        fig.add_annotation(x=x_end, y=0, ax=x_start, ay=0, xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=2, arrowsize=1, arrowcolor=PALETTE["border"], arrowwidth=2)
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=PALETTE["bg"], paper_bgcolor=PALETTE["bg"],
        xaxis=dict(visible=False, range=[0, 11.5]),
        yaxis=dict(visible=False, range=[-1, 1]),
    )
    return fig


def build_accuracy_chart():
    x = ["120 min out", "90 min out", "60 min out", "30 min out", "15 min out", "5 min out"]
    y = [11.4, 9.2, 7.1, 5.3, 3.6, 2.1]
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers",
                                line=dict(color=PALETTE["accent"], width=3),
                                fill="tozeroy", fillcolor="rgba(79,209,197,0.15)"))
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["muted"]),
        yaxis=dict(title="Minutes", gridcolor=PALETTE["border"]),
        xaxis=dict(gridcolor=PALETTE["border"]),
    )
    return fig


def build_causes_chart():
    labels = ["Congestion", "Preceding train", "Weather", "Infrastructure", "Crew & ops"]
    values = [34, 26, 15, 14, 11]
    colors = [PALETTE["accent"], PALETTE["caution"], PALETTE["danger"], PALETTE["muted"], PALETTE["clear"]]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55, marker=dict(colors=colors)))
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["muted"]), legend=dict(orientation="h", y=-0.15),
    )
    return fig


def build_zones_chart():
    labels = ["Western", "Eastern", "Northern", "South Central", "South Eastern", "Central"]
    values = [81, 74, 69, 77, 72, 80]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=PALETTE["accent"]))
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["muted"]),
        yaxis=dict(range=[0, 100], gridcolor=PALETTE["border"]),
        xaxis=dict(gridcolor=PALETTE["border"]),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🚆 RailPulse")
    st.caption("Dynamic ETA forecasting for Indian Railways coaching trains.")
    if st.button("🔄 Simulate next update", use_container_width=True):
        tick()
    st.caption(
        "Advances every train's delay and re-weights its contributing factors, "
        "the way a new GPS or signal reading would in production."
    )
    st.caption(f"Last simulated: {st.session_state.last_tick.strftime('%H:%M:%S')}")
    st.markdown("---")
    st.caption(
        "Built for SIH 2026 problem statement SIH26028 (Ministry of Railways). "
        "Uses simulated data to demonstrate the approach."
    )

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------

statuses = [status_of(t["delay"]) for t in trains]
on_time = sum(1 for s in statuses if s[0] == "clear")
caution_n = sum(1 for s in statuses if s[0] == "caution")
major_n = sum(1 for s in statuses if s[0] == "danger")
network_emoji = "🔴" if major_n > 0 else ("🟡" if caution_n > 0 else "🟢")

st.markdown(f"## {network_emoji} RailPulse")
st.caption("Live running status and dynamically updated arrival predictions for coaching trains across the network.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Trains in this demo", len(trains))
k2.metric("On time now", on_time)
k3.metric("Active major-delay alerts", major_n)
k4.metric("Backtested MAE, 90 days (simulated)", "4.2 min")

tab1, tab2, tab3, tab4 = st.tabs(["Live board", "Train detail", "System & analytics", "API & integration"])

# ---------------------------------------------------------------------------
# Tab 1: Live board
# ---------------------------------------------------------------------------

with tab1:
    rows = []
    for t in trains:
        next_idx = t["at_station"] + 1
        next_station = t["stations"][next_idx]
        sched_clock, sched_day = to_clock_and_day(t["start_clock"], next_station["sched"])
        pred_clock, pred_day = to_clock_and_day(t["start_clock"], next_station["sched"] + t["delay"])
        _, label, emoji = status_of(t["delay"])
        rows.append({
            "Train": t["no"],
            "Name": t["name"],
            "Destination": t["to"],
            "Next stop": next_station["name"],
            "Scheduled": sched_clock + (f" +{sched_day}d" if sched_day > 0 else ""),
            "Predicted": pred_clock + (f" +{pred_day}d" if pred_day > 0 else ""),
            "Delay": f"{'+' if t['delay'] >= 0 else ''}{round(t['delay'])} min",
            "Status": f"{emoji} {label}",
        })
    st.subheader("Departure board")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Click \"Simulate next update\" in the sidebar to advance the predictions, "
        "the way new signal, GPS and weather data would in production."
    )

# ---------------------------------------------------------------------------
# Tab 2: Train detail
# ---------------------------------------------------------------------------

with tab2:
    labels = [f"{t['no']}/{t['name']}" for t in trains]
    default_idx = next(i for i, t in enumerate(trains) if t["no"] == "12809")
    choice = st.selectbox("Train", labels, index=default_idx)
    selected_no = choice.split("/")[0]
    t = next(x for x in trains if x["no"] == selected_no)

    next_idx = t["at_station"] + 1
    _, status_label, status_emoji = status_of(t["delay"])
    dist_to_next = t["stations"][next_idx]["km"] - t["stations"][t["at_station"]]["km"]

    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown(f"#### {t['no']}  {t['name']}")
        st.caption(t["zone"])
        st.caption(f"{t['from']} \u2192 {t['to']}")
    with colB:
        st.markdown(f"**{status_emoji} {status_label}**")
        st.caption(f"Running at about {t['speed']} km/h, {dist_to_next} km from {t['stations'][next_idx]['name']}")

    st.plotly_chart(build_route_figure(t), use_container_width=True, config={"displayModeBar": False})

    cards = st.columns(len(t["stations"]))
    for i, (s, col) in enumerate(zip(t["stations"], cards)):
        with col:
            if i <= t["at_station"]:
                clock, day = to_clock_and_day(t["start_clock"], s["sched"])
                st.markdown(f"**{s['name']}**")
                st.caption(clock + (f" +{day}d" if day > 0 else ""))
                st.caption("Departed")
            else:
                stations_ahead = i - t["at_station"]
                conf = confidence_minutes(stations_ahead, t)
                clock, day = to_clock_and_day(t["start_clock"], s["sched"] + t["delay"])
                st.markdown(f"**{s['name']}**")
                st.markdown(clock + (f" +{day}d" if day > 0 else ""))
                st.caption(f"\u00B1 {conf} min")

    st.markdown("#### Why this prediction")
    st.caption("Live contributing factors behind the current ETA, reweighted as new data arrives.")
    for k, v in t["factors"].items():
        bar_col, val_col = st.columns([5, 1])
        with bar_col:
            st.caption(FACTOR_LABELS[k])
            st.progress(v / 100)
        with val_col:
            st.markdown(f"**{v:.1f}%**")

    next_conf = confidence_minutes(1, t)
    next_clock, next_day = to_clock_and_day(t["start_clock"], t["stations"][next_idx]["sched"] + t["delay"])
    st.markdown("---")
    st.caption("Next stop")
    st.markdown(f"### {t['stations'][next_idx]['name']}")
    st.markdown(f"## {next_clock}{f' +{next_day}d' if next_day > 0 else ''}")
    st.caption(f"\u00B1 {next_conf} min")

# ---------------------------------------------------------------------------
# Tab 3: System & analytics
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("How the prediction engine works")
    st.caption("Figures on this page are illustrative, generated for this prototype.")
    st.plotly_chart(build_architecture_figure(), use_container_width=True, config={"displayModeBar": False})

    a1, a2, a3, a4, a5 = st.columns(5)
    with a1:
        st.markdown("**Data sources**")
        st.caption("Live GPS pings, signal and block occupancy, weather feeds, historical running-time and delay logs, and crew or maintenance block schedules.")
    with a2:
        st.markdown("**Feature engineering**")
        st.caption("Turns raw feeds into model inputs: current delay, distance to next station, section averages, downstream congestion and time-of-day seasonality.")
    with a3:
        st.markdown("**Prediction engine**")
        st.caption("A gradient-boosted regression model trained on historical running times, corrected in real time as new location and delay signals arrive.")
    with a4:
        st.markdown("**Serving APIs**")
        st.caption("Predictions are recomputed on every new event and exposed through REST and WebSocket endpoints with low latency.")
    with a5:
        st.markdown("**Consumers**")
        st.caption("The same predictions power passenger apps, station display boards and control room dashboards.")

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Prediction accuracy by lead time**")
        st.plotly_chart(build_accuracy_chart(), use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.markdown("**Network delay causes**")
        st.plotly_chart(build_causes_chart(), use_container_width=True, config={"displayModeBar": False})
    with c3:
        st.markdown("**On-time performance by zone**")
        st.plotly_chart(build_zones_chart(), use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Tab 4: API & integration
# ---------------------------------------------------------------------------

with tab4:
    st.subheader("API & integration")
    st.caption("Sample endpoints for the mobile app, station displays and control room dashboards described in the problem statement.")

    st.markdown("**GET** `/api/v1/trains/{train_no}/eta`")
    st.caption("Current predicted arrival time at every upcoming station for one train.")
    st.code("""{
  "train_no": "12809",
  "generated_at": "2026-09-01T10:42:00+05:30",
  "current_delay_min": 34,
  "next_station": {
    "code": "BSP",
    "name": "Bilaspur",
    "eta": "2026-09-01T11:58:00+05:30",
    "confidence_min": 6
  },
  "upcoming": [
    {
      "code": "HWH",
      "name": "Howrah",
      "eta": "2026-09-02T13:15:00+05:30",
      "confidence_min": 22
    }
  ]
}""", language="json")

    st.markdown("**GET** `/api/v1/trains/{train_no}/factors`")
    st.caption("The contributing-factor breakdown behind the current prediction.")
    st.code("""{
  "train_no": "12809",
  "factors": [
    { "name": "historical_running_time", "weight_pct": 38.2 },
    { "name": "live_congestion", "weight_pct": 27.5 },
    { "name": "preceding_train_delay", "weight_pct": 18.9 },
    { "name": "weather", "weight_pct": 9.7 },
    { "name": "speed_restriction", "weight_pct": 5.7 }
  ]
}""", language="json")

    st.markdown("**GET** `/api/v1/network/status`")
    st.caption("Network-wide summary for a control room dashboard.")
    st.code("""{
  "trains_tracked": 6142,
  "on_time_pct": 76.4,
  "major_delay_count": 118,
  "updated_at": "2026-09-01T10:42:00+05:30"
}""", language="json")

    st.markdown("**WS** `wss://api.example.org/v1/stream/trains/{train_no}`")
    st.caption("Subscribe for push updates whenever a new prediction is computed, instead of polling.")
    st.code("""{
  "type": "eta_update",
  "train_no": "12809",
  "next_station": "BSP",
  "eta": "2026-09-01T11:58:00+05:30",
  "delay_min": 34
}""", language="json")

    st.markdown("**Example client call**")
    st.code("""fetch('/api/v1/trains/12809/eta')
  .then(r => r.json())
  .then(data => updateDisplay(data));""", language="javascript")
