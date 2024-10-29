import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Helper functions for visualization
def create_metrics_dashboard(df):
    """Create a metrics dashboard with key KPIs"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_leads = df['leads'].mean()
        st.metric(
            "Avg Monthly Leads",
            f"{avg_leads:,.0f}",
            f"{(df['leads'].iloc[-1] - avg_leads) / avg_leads:,.1%}"
        )
    
    with col2:
        avg_appointments = df['appointments'].mean()
        st.metric(
            "Avg Monthly Appointments",
            f"{avg_appointments:,.0f}",
            f"{(df['appointments'].iloc[-1] - avg_appointments) / avg_appointments:,.1%}"
        )
    
    with col3:
        conversion_rate = (df['appointments'] / df['leads']).mean() * 100
        last_conversion = (df['appointments'].iloc[-1] / df['leads'].iloc[-1]) * 100
        st.metric(
            "Lead-to-Appointment Rate",
            f"{conversion_rate:.1f}%",
            f"{last_conversion - conversion_rate:.1f}%"
        )
    
    with col4:
        closing_rate = (df['closings'] / df['appointments']).mean() * 100
        last_closing_rate = (df['closings'].iloc[-1] / df['appointments'].iloc[-1]) * 100
        st.metric(
            "Appointment-to-Close Rate",
            f"{closing_rate:.1f}%",
            f"{last_closing_rate - closing_rate:.1f}%"
        )

def plot_interactive_trends(df):
    """Create interactive trend visualization with Plotly"""
    metrics = st.multiselect(
        "Select metrics to display:",
        options=['leads', 'appointments', 'closings'],
        default=['leads', 'appointments', 'closings']
    )
    
    fig = go.Figure()
    colors = {'leads': '#1f77b4', 'appointments': '#ff7f0e', 'closings': '#2ca02c'}
    
    for metric in metrics:
        fig.add_trace(
            go.Scatter(
                x=df['month'],
                y=df[metric],
                name=metric.capitalize(),
                line=dict(color=colors[metric]),
                hovertemplate="Date: %{x}<br>" +
                             f"{metric.capitalize()}: %{{y:,.0f}}<br>" +
                             "<extra></extra>"
            )
        )
    
    fig.update_layout(
        title="Historical Trends",
        xaxis_title="Month",
        yaxis_title="Count",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_conversion_funnel(df):
    """Create an interactive conversion funnel visualization"""
    avg_leads = df['leads'].mean()
    avg_appointments = df['appointments'].mean()
    avg_closings = df['closings'].mean()
    
    fig = go.Figure(go.Funnel(
        y=['Leads', 'Appointments', 'Closings'],
        x=[avg_leads, avg_appointments, avg_closings],
        textposition="inside",
        textinfo="value+percent initial",
        opacity=0.65,
        marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ))
    
    fig.update_layout(title="Conversion Funnel")
    st.plotly_chart(fig, use_container_width=True)

# Title of the app
st.title("Lead, Appointment, and Closing Stats Forecaster")

# Input section for user manual input
st.header("Input Your Data Manually")
num_leads = st.number_input("Number of Leads:", min_value=0, value=100)
num_appointments = st.number_input("Number of Appointments:", min_value=0, value=50)
num_closings = st.number_input("Number of Closings:", min_value=0, value=25)
average_revenue_per_closing = st.number_input("Average Revenue Per Closing ($):", 
                                            min_value=0.0, 
                                            value=10000.0)

# Create sample data
sample_data = {
    'month': ['2023-01-01', '2023-02-01', '2023-03-01'],
    'leads': [100, 120, 110],
    'appointments': [50, 60, 55],
    'closings': [25, 30, 28]
}
sample_df = pd.DataFrame(sample_data)
csv_file = sample_df.to_csv(index=False).encode('utf-8')

# Download button for sample data
st.download_button(
    label="Download Example CSV",
    data=csv_file,
    file_name='example_data.csv',
    mime='text/csv',
)

# File upload section
st.header("Upload Historical Data")
st.info("Please upload a CSV file with the following columns: 'month', 'leads', 'appointments', 'closings'")
st.write("**Example CSV Format:**")
st.write(sample_df)

uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Read and process the data
        historical_data = pd.read_csv(uploaded_file)
        historical_data['month'] = pd.to_datetime(historical_data['month'])
        
        # Data overview in expandable section
        with st.expander("View Raw Data"):
            st.dataframe(historical_data)
        
        # Interactive date range filter
        st.sidebar.header("Data Filters")
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(
                historical_data['month'].min(),
                historical_data['month'].max()
            )
        )
        
        # Filter data based on date range
        mask = (historical_data['month'].dt.date >= date_range[0]) & \
               (historical_data['month'].dt.date <= date_range[1])
        filtered_data = historical_data.loc[mask]
        
        # Display metrics dashboard
        st.header("Key Metrics Dashboard")
        create_metrics_dashboard(filtered_data)
        
        # Interactive trend visualization
        st.header("Historical Trends")
        plot_interactive_trends(filtered_data)
        
        # Conversion funnel
        st.header("Conversion Funnel")
        plot_conversion_funnel(filtered_data)
        
        # Forecast section
        st.header("Forecast Analysis")
        
        # Fit model and make predictions
        x = filtered_data[['leads', 'appointments']]
        y = filtered_data['closings']
        model = LinearRegression()
        model.fit(x, y)
        
        input_data = np.array([[num_leads, num_appointments]]).reshape(1, -1)
        forecasted_closings = max(0, model.predict(input_data)[0])
        forecasted_revenue = forecasted_closings * average_revenue_per_closing
        
        # Display forecast results
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Forecasted Closings",
                f"{forecasted_closings:.1f}",
                f"{(forecasted_closings - filtered_data['closings'].mean()) / filtered_data['closings'].mean():,.1%}"
            )
        with col2:
            st.metric(
                "Forecasted Revenue",
                f"${forecasted_revenue:,.2f}",
                f"${forecasted_revenue - (filtered_data['closings'].mean() * average_revenue_per_closing):,.2f}"
            )
        
        # Model quality metrics
        with st.expander("Model Quality Metrics"):
            r2_score = model.score(x, y)
            st.write(f"R² Score: {r2_score:.3f}")
            st.write(f"Coefficients:")
            st.write("- Leads Impact: {:.3f}".format(model.coef_[0]))
            st.write("- Appointments Impact: {:.3f}".format(model.coef_[1]))
            
            if r2_score < 0.5:
                st.warning("Warning: Model fit is poor. Predictions may be unreliable.")
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.write("Please ensure your CSV file is properly formatted and try again.")
