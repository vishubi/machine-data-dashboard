import pandas as pd
import streamlit as st
import plotly.express as px

# Load dataset
st.title("Machine Data Interactive Dashboard")

try:
    df = pd.read_csv("synthetic_machine_data_lstm.csv", parse_dates=["Timestamp"])
    st.write("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("Error: Dataset file not found. Please check the file path.")
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Time-Series Analysis"])

if page == "Overview":
    st.header("Dataset Overview")
    st.write(df.head())
    st.write(f"Total Rows: {df.shape[0]}")
    st.write(f"Total Columns: {df.shape[1]}")

if page == "Time-Series Analysis":
    st.header("Time-Series Data")
    selected_feature = st.selectbox("Select a feature to plot over time:", [col for col in df.columns if col != "Timestamp"])
    fig = px.line(df, x="Timestamp", y=selected_feature, title=f"{selected_feature} Over Time")
    st.plotly_chart(fig)
