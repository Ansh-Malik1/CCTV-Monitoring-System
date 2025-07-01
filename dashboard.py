import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("🛡️ CCTV Factory Monitoring Dashboard")

# Load violations if any
def load_violations():
    try:
        with open("violations.csv") as f:
            return pd.read_csv(f)
    except FileNotFoundError:
        return pd.DataFrame(columns=["frame", "zone", "x", "y"])

df = load_violations()

# Show violation table
st.subheader("🚨 Violation Log")
st.dataframe(df, use_container_width=True)

# Show zone layout
st.subheader("📍 Restricted Zones Map")
with open("zones.json") as f:
    zones = json.load(f)

for z in zones["zones"]:
    st.markdown(f"**{z['name']}** — Coords: {z['polygon']}")
