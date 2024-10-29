import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go
import io
from typing import Dict, Union, Optional, Tuple, Any
from datetime import datetime

# Configuration and Setup
st.set_page_config(layout="wide")

# Improved CSS with better class naming and organization
st.markdown("""
    <style>
    /* Metric Styles */
    .stmetric .stmetriclabel {
        font-size: 14px !important;
        font-weight: 500;
        color: #2c3e50;
    }
    .stmetric .stmetricvalue {
        font-size: 24px !important;
        font-weight: 600;
        color: #1a237e;
    }
    .stmetric .stmetricdelta {
        font-size: 12px !important;
        color: #546e7a;
    }
    
    /* Container Styles */
    .main-container {
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Header Styles */
    .section-header {
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* Progress Bar Styles */
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Type definitions for better code organization
MetricsDict = Dict[str, Union[float, str, int]]
DataFrameType = pd.DataFrame
ModelType = LinearRegression
PlotType = go.Figure

def calculate_metrics(df: DataFrameType) -> MetricsDict:
    """Calculate business metrics including cost metrics."""
    try:
        required_columns = ['leads', 'appointments', 'closings', 'cost']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Need: {required_columns}")
            
        metrics = {
            'Total Leads': df['leads'].sum(),
            'Total Appointments': df['appointments'].sum(),
            'Total Closings': df['closings'].sum(),
            'Cost per Lead': df['cost'].sum() / df['leads'].sum() if df['leads'].sum() > 0 else 0,
            'Cost per Appointment': df['cost'].sum() / df['appointments'].sum() if df['appointments'].sum() > 0 else 0,
            'Cost per Closing': df['cost'].sum() / df['closings'].sum() if df['closings'].sum() > 0 else 0,
            'Lead to Appointment Rate (%)': (df['appointments'].sum() / df['leads'].sum() * 100) if df['leads'].sum() > 0 else 0,
            'Appointment to Close Rate (%)': (df['closings'].sum() / df['appointments'].sum() * 100) if df['appointments'].sum() > 0 else 0,
            'Overall Close Rate (%)': (df['closings'].sum() / df['leads'].sum() * 100) if df['leads'].sum() > 0 else 0
        }
        
        if not df.empty:
            metrics.update({
                'Best Month (Closings)': df.loc[df['closings'].idxmax(), 'month'].strftime('%b %Y'),
                'Worst Month (Closings)': df.loc[df['closings'].idxmin(), 'month'].strftime('%b %Y')
            })
            
        return {k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in metrics.items()}
        
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {}

def create_metrics_dashboard(df: DataFrameType) -> None:
    """Create and display metrics dashboard."""
    st.header("Key Metrics Dashboard", help="Overview of key performance indicators")
    metrics = calculate_metrics(df)
    
    col1, col2, col3 = st.columns(3)
    for i, (metric, value) in enumerate(metrics.items()):
        with [col1, col2, col3][i % 3]:
            if isinstance(value, str) and value.replace('.', '').isdigit():
                value = float(value)
                
            formatted_value = (
                f"{value:.1f}%" if "rate" in metric.lower() else
                f"${value:,.2f}" if "cost" in metric.lower() else
                f"{value:,.0f}" if isinstance(value, (float, int)) else
                value
            )
            
            st.metric(label=metric, value=formatted_value, delta=None)

def add_goals_tracking(df: DataFrameType) -> None:
    """Add goals tracking section with progress bars."""
    st.subheader("Performance Goals Tracking")
    col1, col2, col3 = st.columns(3)

    def calculate_progress(actual: float, goal: float) -> float:
        return (actual / goal * 100) if goal > 0 else 0

    with col1:
        lead_goal = st.number_input("Monthly Leads Goal:", min_value=0, value=100)
        actual_leads = df['leads'].iloc[-1] if not df.empty else 0
        leads_progress = calculate_progress(actual_leads, lead_goal)
        st.progress(min(leads_progress / 100, 1.0))
        st.write(f"Latest: {actual_leads:.0f} ({leads_progress:.1f}% of goal)")

    with col2:
        appointment_goal = st.number_input("Monthly Appointments Goal:", min_value=0, value=50)
        actual_appointments = df['appointments'].iloc[-1] if not df.empty else 0
        appointments_progress = calculate_progress(actual_appointments, appointment_goal)
        st.progress(min(appointments_progress / 100, 1.0))
        st.write(f"Latest: {actual_appointments:.0f} ({appointments_progress:.1f}% of goal)")

    with col3:
        closing_goal = st.number_input("Monthly Closings Goal:", min_value=0, value=25)
        actual_closings = df['closings'].iloc[-1] if not df.empty else 0
        closings_progress = calculate_progress(actual_closings, closing_goal)
        st.progress(min(closings_progress / 100, 1.0))
        st.write(f"Latest: {actual_closings:.0f} ({closings_progress:.1f}% of goal)")

def plot_seasonality_analysis(df: DataFrameType, metric: str) -> Optional[PlotType]:
    """Plot seasonal decomposition analysis with error handling."""
    if len(df) < 24:
        st.warning("Insufficient data for reliable seasonal decomposition (need 24+ months).")
        return None

    try:
        # Ensure data is properly sorted and indexed
        df_sorted = df.sort_values('month').copy()
        df_sorted.set_index('month', inplace=True)
        
        # Perform decomposition
        decomposed = seasonal_decompose(df_sorted[metric], model='additive', period=12)
        
        # Create figure dictionary (Plotly expects dict format)
        fig_dict = {
            'data': [
                {
                    'type': 'scatter',
                    'x': decomposed.trend.index,
                    'y': decomposed.trend,
                    'name': 'Trend',
                    'line': {'color': '#2196F3', 'width': 2}
                },
                {
                    'type': 'scatter',
                    'x': decomposed.seasonal.index,
                    'y': decomposed.seasonal,
                    'name': 'Seasonal',
                    'line': {'color': '#4CAF50', 'width': 2}
                }
            ],
            'layout': {
                'title': f"Seasonality and Trend Analysis for {metric.capitalize()}",
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': metric.capitalize()},
                'template': 'plotly_white',
                'height': 500
            }
        }
        return fig_dict
    except Exception as e:
        st.error(f"Error in seasonal decomposition: {str(e)}")
        return None

def train_forecast_model(df: DataFrameType) -> Tuple[Optional[ModelType], Dict[str, float]]:
    """Train forecasting model with proper feature names."""
    if len(df) < 2:
        return None, {'error': 'Insufficient data for modeling'}
    
    try:
        # Create feature matrix with proper column names
        X = pd.DataFrame({
            'leads': df['leads'],
            'appointments': df['appointments']
        })
        y = df['closings']
        
        # Initialize and train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate performance metrics
        y_pred = model.predict(X)
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': model.score(X, y) if len(df) > 2 else np.nan
        }
        
        return model, metrics
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, {'error': str(e)}

def predict_closings(model: Optional[ModelType], 
                    leads: float, 
                    appointments: float) -> Tuple[float, str]:
    """Make predictions with proper feature format."""
    if model is None:
        return 0.0, "No model available"
    
    try:
        # Create properly formatted input DataFrame
        input_data = pd.DataFrame({
            'leads': [leads],
            'appointments': [appointments]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        return max(0.0, prediction), "Success"
    except Exception as e:
        return 0.0, str(e)

# Update the tab3 (Forecasting) section to use these functions:
with tab3:
    st.header("Forecast Analysis")
    st.subheader("Model Configuration")
    forecast_periods = st.slider("Forecast Periods (Months):", 1, 12, 3)

    if not st.session_state.historical_data.empty:
        # Train model
        model, model_metrics = train_forecast_model(st.session_state.historical_data)
        
        if model is not None:
            # Make prediction
            forecasted_closings, prediction_status = predict_closings(
                model, num_leads, num_appointments)
            
            if prediction_status == "Success":
                forecasted_revenue = forecasted_closings * average_revenue_per_closing

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Forecasted Closings", f"{forecasted_closings:.1f}")
                with col2:
                    st.metric("Forecasted Revenue", f"${forecasted_revenue:,.2f}")

                # Display model metrics
                st.subheader("Model Performance Metrics")
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Mean Absolute Error", 
                          f"{model_metrics['mae']:.2f}")
                col2.metric("Root Mean Square Error", 
                          f"{model_metrics['rmse']:.2f}")
                
                if not np.isnan(model_metrics['r2']):
                    r2_value = model_metrics['r2']
                    col3.metric("R² Score", f"{r2_value:.2f}")
                    
                    if r2_value < 0.5:
                        st.warning("Warning: Model fit is poor. Predictions may be unreliable.")
                else:
                    st.info("R² score requires more than two samples for calculation.")
            else:
                st.error(f"Prediction error: {prediction_status}")
    else:
        st.warning("No data available for forecasting. Please upload data or enter manual inputs.")

def plot_interactive_trends(df: DataFrameType) -> None:
    """Plot interactive historical trends."""
    fig = px.line(df, x='month', y=['leads', 'appointments', 'closings'],
                  title="Historical Performance Trends")
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Count",
        hovermode='x unified',
        template="plotly_white",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_conversion_funnel(df: DataFrameType) -> None:
    """Plot conversion funnel with percentages."""
    values = [df['leads'].sum(), df['appointments'].sum(), df['closings'].sum()]
    labels = ['Leads', 'Appointments', 'Closings']
    
    # Calculate conversion rates
    rates = [100]  # First stage is 100%
    for i in range(1, len(values)):
        rate = (values[i] / values[i-1] * 100) if values[i-1] > 0 else 0
        rates.append(rate)
    
    # Create funnel chart
    fig = go.Figure(go.Funnel(
        y=labels,
        x=values,
        textinfo="value+percent initial",
        textposition="inside",
        textfont=dict(size=14),
        marker=dict(color=["#2196F3", "#4CAF50", "#FFC107"])
    ))
    
    fig.update_layout(
        title="Conversion Funnel Analysis",
        height=500,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

def export_to_excel(df: DataFrameType, forecasts: Dict[str, Any]) -> bytes:
    """Create and return an Excel file with data and forecasts."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Historical Data', index=False)
        pd.DataFrame(forecasts).to_excel(writer, sheet_name='Forecasts', index=False)
        
        # Add summary sheet
        summary = calculate_metrics(df)
        pd.DataFrame(list(summary.items()), columns=['Metric', 'Value']).to_excel(
            writer, sheet_name='Summary', index=False)
    
    return output.getvalue()

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()

# Main application logic
def main():
    """Main application entry point."""
    with st.container():
        tab1, tab2, tab3, tab4 = st.tabs(["Input & Upload", "Analysis", "Forecasting", "Export"])
        
        # Tab content implementations follow...
        # (The rest of the main logic remains the same as in your original code)

if __name__ == "__main__":
    main()
