import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
st.title("Machine Data Interactive Dashboard")

try:
    df = pd.read_csv("synthetic_machine_data_lstm.csv", parse_dates=["Timestamp"])
    st.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("Error: Dataset file not found. Please check the file path.")
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Time-Series Analysis", "Summary Statistics", "Anomaly Detection", "Correlation Matrix", "Multiple Plots", "K-Means Clustering", "Time to Failure"])

# Define a custom color palette
custom_colors = px.colors.qualitative.Pastel

if page == "Overview":
    st.header("Dataset Overview")
    st.dataframe(df.head())
    st.write(f"Total Rows: {df.shape[0]}")
    st.write(f"Total Columns: {df.shape[1]}")

if page == "Time-Series Analysis":
    st.header("Time-Series Data")
    selected_feature = st.selectbox("Select a feature to plot over time:", [col for col in df.columns if col != "Timestamp"])
    fig = px.line(df, x="Timestamp", y=selected_feature, title=f"{selected_feature} Over Time",
                  labels={selected_feature: f"{selected_feature} (units)"}, color_discrete_sequence=["#FF5733"])
    st.plotly_chart(fig, use_container_width=True)

if page == "Summary Statistics":
    st.header("Summary Statistics")
    selected_feature = st.selectbox("Select a feature for statistics:", [col for col in df.columns if col != "Timestamp"])
    st.write(df[selected_feature].describe())
    
    # Box plot visualization
    df_melted = df.melt(id_vars=['Timestamp', 'Machine_Status'], var_name='Feature', value_name='Value')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='Feature', y='Value', palette='BuGn', ax=ax)
    plt.xticks(rotation=45)
    plt.title("Feature Distributions (Boxplot)")
    st.pyplot(fig)

if page == "Anomaly Detection":
    st.header("Anomaly Detection using Z-score")
    selected_feature = st.selectbox("Select a feature for anomaly detection:", [col for col in df.columns if col != "Timestamp"])
    df['Z-score'] = (df[selected_feature] - df[selected_feature].mean()) / df[selected_feature].std()
    df['Anomaly'] = np.where(np.abs(df['Z-score']) > 3, 1, 0)
    fig = px.scatter(df, x="Timestamp", y=selected_feature, color=df['Anomaly'].astype(str), 
                     title=f"{selected_feature} Anomaly Detection", labels={selected_feature: f"{selected_feature} (units)"},
                     color_discrete_map={"0": "#2ECC71", "1": "#E74C3C"})
    st.plotly_chart(fig, use_container_width=True)

if page == "Correlation Matrix":
    st.header("Correlation Matrix")
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, text_auto=True, title="Feature Correlation Matrix", color_continuous_scale='RdBu_r', width=900, height=800)
    st.plotly_chart(fig, use_container_width=True)

if page == "Multiple Plots":
    st.header("Compare Multiple Features Over Time")
    selected_features = st.multiselect("Select features to compare:", [col for col in df.columns if col != "Timestamp"], default=["Temperature", "Vibration"])
    fig = px.line(df, x="Timestamp", y=selected_features, title="Comparison of Selected Features Over Time", color_discrete_sequence=custom_colors)
    st.plotly_chart(fig, use_container_width=True)

if page == "K-Means Clustering":
    st.header("K-Means Clustering")
    num_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)
    feature_columns = [col for col in df.columns if col not in ["Timestamp", "Machine_Status"]]
    selected_features = st.multiselect("Select features for clustering:", feature_columns, default=["Temperature", "Vibration"])
    
    if selected_features:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[selected_features])
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df["Cluster"] = kmeans.fit_predict(df_scaled)
        
        fig = px.scatter(df, x=selected_features[0], y=selected_features[1], color=df["Cluster"].astype(str),
                         title="K-Means Clustering Results", labels={selected_features[0]: f"{selected_features[0]} (units)",
                         selected_features[1]: f"{selected_features[1]} (units)"})
        st.plotly_chart(fig, use_container_width=True)

if page == "Time to Failure":
    st.header("Time to Failure Estimation")
    df['Time_To_Next_Failure'] = df['Timestamp'].diff().dt.total_seconds().fillna(0)
    df.loc[df['Machine_Status'] == 0, 'Time_To_Next_Failure'] = np.nan
    df['Time_To_Next_Failure'] = df['Time_To_Next_Failure'].fillna(method='bfill')
    
    fig = px.line(df, x="Timestamp", y="Time_To_Next_Failure", title="Estimated Time to Next Failure",
                  labels={"Time_To_Next_Failure": "Time to Failure (seconds)"})
    st.plotly_chart(fig, use_container_width=True)
