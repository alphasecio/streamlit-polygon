import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go
import io

# Set page layout
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .stmetric .stmetriclabel {font-size: 14px !important;}
    .stmetric .stmetricvalue {font-size: 24px !important;}
    .stmetric .stmetricdelta {font-size: 12px !important;}
    </style>
    """, unsafe_allow_html=True)

# Initialize historical_data if not already done
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()

# Callback function to refresh metrics after inputs change
def update_metrics():
    # Triggers recalculation and update of metrics by updating a key value
    st.session_state["metrics_updated"] = True

# Functions to calculate and display metrics
def calculate_metrics(df):
    """Calculate business metrics including cost metrics."""
    try:
        metrics = {
            'Total Leads': df['leads'].sum(),
            'Total Appointments': df['appointments'].sum(),
            'Total Closings': df['closings'].sum(),
            'Cost per Lead': (df['cost'].sum() / df['leads'].sum() if df['leads'].sum() > 0 else 0),
            'Cost per Appointment': (df['cost'].sum() / df['appointments'].sum() if df['appointments'].sum() > 0 else 0),
            'Cost per Closing': (df['cost'].sum() / df['closings'].sum() if df['closings'].sum() > 0 else 0),
            'Lead to Appointment Rate': (df['appointments'].sum() / df['leads'].sum() * 100 if df['leads'].sum() > 0 else 0),
            'Appointment to Close Rate': (df['closings'].sum() / df['appointments'].sum() * 100 if df['appointments'].sum() > 0 else 0),
            'Overall Close Rate': (df['closings'].sum() / df['leads'].sum() * 100 if df['leads'].sum() > 0 else 0),
            'Best Month (Closings)': df.loc[df['closings'].idxmax(), 'month'].strftime('%b %y') if df['closings'].any() else 'N/A',
            'Worst Month (Closings)': df.loc[df['closings'].idxmin(), 'month'].strftime('%b %y') if df['closings'].any() else 'N/A',
        }
        return {k: (v if pd.notna(v) and isinstance(v, (int, float)) else 'N/A') for k, v in metrics.items()}
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {}

def create_metrics_dashboard(df):
    """Create metrics dashboard for display."""
    metrics = calculate_metrics(df)
    col1, col2, col3 = st.columns(3)

    for i, (metric, value) in enumerate(metrics.items()):
        with [col1, col2, col3][i % 3]:
            formatted_value = f"{value:.1f}%" if "rate" in metric else f"${value:.2f}" if "cost per" in metric else f"{value:,.0f}" if isinstance(value, (float, int)) else value
            st.metric(metric, formatted_value)

# Input data section with callbacks to refresh metrics on input change
st.header("Input Your Data")
col1, col2 = st.columns(2)

with col1:
    num_leads = st.number_input("Number of Leads:", min_value=0, value=100, on_change=update_metrics, key="leads_input")
    num_appointments = st.number_input("Number of Appointments:", min_value=0, value=50, on_change=update_metrics, key="appointments_input")

with col2:
    num_closings = st.number_input("Number of Closings:", min_value=0, value=25, on_change=update_metrics, key="closings_input")
    average_revenue_per_closing = st.number_input("Average Revenue Per Closing ($):", min_value=0.0, value=10000.0, on_change=update_metrics, key="revenue_per_closing")
    cost = st.number_input("Total Cost (Enter 0 if none):", min_value=0.0, value=0.0, on_change=update_metrics, key="cost_input")

# Handle file uploads
uploaded_file = st.file_uploader("Upload Your Data File", type=["csv", "xlsx"], on_change=update_metrics)

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.historical_data = pd.read_csv(uploaded_file)
        else:
            st.session_state.historical_data = pd.read_excel(uploaded_file)

        st.session_state.historical_data['month'] = pd.to_datetime(st.session_state.historical_data['month'], errors='coerce')
        st.session_state.historical_data.dropna(subset=['month'], inplace=True)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# Add manual input to historical data if available
if not st.session_state.historical_data.empty:
    new_row = pd.DataFrame({
        'month': [pd.Timestamp('now')],
        'leads': [st.session_state["leads_input"]],
        'appointments': [st.session_state["appointments_input"]],
        'closings': [st.session_state["closings_input"]],
        'cost': [st.session_state["cost_input"]]
    })
    st.session_state.historical_data = pd.concat([st.session_state.historical_data, new_row], ignore_index=True)

st.write("Combined Data Overview:")
st.dataframe(st.session_state.historical_data)

# Trigger dashboard refresh based on metrics_updated key
if "metrics_updated" in st.session_state:
    create_metrics_dashboard(st.session_state.historical_data)
