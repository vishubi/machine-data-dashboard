import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
st.title("Machine Data Monitoring Dashboard")
st.write("Loading dataset...")

try:
    df = pd.read_csv("synthetic_machine_data_lstm.csv", parse_dates=["Timestamp"])
    st.write("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("Error: Dataset file not found. Please check the file path.")
    st.stop()

# Ensure all columns are numeric where necessary
columns = ["Temperature", "Vibration", "Pressure", "Humidity", "Load", "Acoustic_Noise"]
df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# Ensure Machine_Status exists
df['Machine_Status'] = df.get('Machine_Status', np.zeros(len(df)))

# Feature Engineering
st.write("Performing feature engineering...")
df['Temp_Vib_Interaction'] = df['Temperature'] * df['Vibration']
df['Load_Pressure_Ratio'] = df['Load'] / (df['Pressure'] + 1e-6)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df[columns])
poly_feature_names = poly.get_feature_names_out(columns)
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
df = pd.concat([df, df_poly], axis=1)

st.write("Feature engineering completed!")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Time-Series Analysis", "Anomaly Detection", "Summary Statistics"])

if page == "Overview":
    st.header("Dataset Overview")
    st.write(df.head())
    st.write(f"Total Rows: {df.shape[0]}")
    st.write(f"Total Columns: {df.shape[1]}")

if page == "Time-Series Analysis":
    st.header("Time-Series Data")
    selected_feature = st.selectbox("Select a feature to plot over time:", columns)
    fig = px.line(df, x="Timestamp", y=selected_feature, title=f"{selected_feature} Over Time")
    st.plotly_chart(fig)
    
    st.subheader("Temperature vs Vibration Analysis")
    if "Machine_Status" in df.columns:
        fig_scatter = px.scatter(df, x="Temperature", y="Vibration", color=df["Machine_Status"].astype(str), title="Temperature vs Vibration Colored by Machine Status")
        st.plotly_chart(fig_scatter)
    else:
        st.warning("Machine Status column is missing.")

if page == "Anomaly Detection":
    st.header("Anomaly Detection (Z-score)")
    anomaly_feature = st.selectbox("Select a feature for anomaly detection:", columns)
    if f'{anomaly_feature}_anomaly' in df.columns:
        fig_anomaly = px.scatter(df, x="Timestamp", y=anomaly_feature, color=df[f'{anomaly_feature}_anomaly'].astype(str), title=f"{anomaly_feature} Anomalies Over Time")
        st.plotly_chart(fig_anomaly)
    else:
        st.warning("No anomaly data available for this feature.")

if page == "Summary Statistics":
    st.header("Summary Statistics")
    st.write(df.describe())