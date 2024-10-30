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

# Configuration and setup
st.set_page_config(layout="wide")

# Improved CSS for styling
st.markdown("""
    <style>
    /* Metric styling */
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

    /* Container styling */
    .main-container {
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Header styles */
    .section-header {
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
        font-size: 20px;
        color: #0f3460;
    }

    /* Progress bar styles */
    .stprogress > div > div {
        background-color: #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Type definitions
MetricsDict = Dict[str, Union[float, str, int]]
DataFrameType = pd.DataFrame
ModelType = LinearRegression

def generate_sample_data() -> pd.DataFrame:
    """Generate sample data for demonstration."""
    sample_data = {
        'month': pd.date_range(start='2023-01-01', periods=12, freq='ME'),  # Fixed frequency
        'leads': np.random.randint(80, 120, 12),
        'appointments': np.random.randint(40, 60, 12),
        'closings': np.random.randint(20, 30, 12),
        'cost': np.random.randint(5000, 8000, 12)
    }
    return pd.DataFrame(sample_data)

def create_excel_template() -> bytes:
    """Create an Excel template file with sample data."""
    sample_df = generate_sample_data()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        sample_df.to_excel(writer, index=False)
    return output.getvalue()

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
            'Cost per Lead': (df['cost'].sum() / df['leads'].sum() if df['leads'].sum() > 0 else 0),
            'Cost per Appointment': (df['cost'].sum() / df['appointments'].sum() if df['appointments'].sum() > 0 else 0),
            'Cost per Closing': (df['cost'].sum() / df['closings'].sum() if df['closings'].sum() > 0 else 0),
            'Lead to Appointment Rate (%)': (df['appointments'].sum() / df['leads'].sum() * 100 if df['leads'].sum() > 0 else 0),
            'Appointment to Close Rate (%)': (df['closings'].sum() / df['appointments'].sum() * 100 if df['appointments'].sum() > 0 else 0),
            'Overall Close Rate (%)': (df['closings'].sum() / df['leads'].sum() * 100 if df['leads'].sum() > 0 else 0)
        }

        if not df.empty:
            metrics.update({
                'Best Month (Closings)': df.loc[df['closings'].idxmax(), 'month'].strftime('%b %y'),
                'Worst Month (Closings)': df.loc[df['closings'].idxmin(), 'month'].strftime('%b %y')
            })

        return metrics

    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {}

def create_metrics_dashboard(df: DataFrameType) -> None:
    """Create and display metrics dashboard."""
    st.header("Key Metrics Dashboard")

    metrics = calculate_metrics(df)
    col1, col2, col3 = st.columns(3)

    for i, (metric, value) in enumerate(metrics.items()):
        with [col1, col2, col3][i % 3]:
            formatted_value = (
                f"{value:.1f}%" if "rate" in metric.lower() else
                f"${value:,.2f}" if "cost" in metric.lower() else
                f"{value:,.0f}" if isinstance(value, (float, int)) else
                value
            )
            st.metric(label=metric, value=formatted_value)

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

def plot_seasonality_analysis(df: DataFrameType, metric: str) -> Optional[go.Figure]:
    """Plot seasonal decomposition analysis."""
    if len(df) < 24:
        st.warning("Not enough data for seasonal decomposition. Please provide at least 24 observations.")
        return None

    try:
        decomposed = seasonal_decompose(df.set_index('month')[metric], model='additive', period=12)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=decomposed.trend.index, y=decomposed.trend, name='Trend'))
        fig.add_trace(go.Scatter(x=decomposed.seasonal.index, y=decomposed.seasonal, name='Seasonal'))
        fig.update_layout(title=f"Seasonality and Trend Analysis for {metric.capitalize()}", 
                         xaxis_title="Date", 
                         yaxis_title=metric.capitalize())
        return fig
    except Exception as e:
        st.error(f"Error in seasonal decomposition: {str(e)}")
        return None

def plot_interactive_trends(df: DataFrameType) -> None:
    """Plot interactive historical trends."""
    fig = px.line(df, x='month', y=['leads', 'appointments', 'closings'], 
                  title="Historical Performance Trends")
    fig.update_layout(xaxis_title="Month", yaxis_title="Values")
    st.plotly_chart(fig, use_container_width=True)

def plot_conversion_funnel(df: DataFrameType) -> None:
    """Plot conversion funnel with percentages."""
    values = [df['leads'].sum(), df['appointments'].sum(), df['closings'].sum()]
    labels = ['Leads', 'Appointments', 'Closings']

    fig = go.Figure(go.Funnel(
        y=labels,
        x=values,
        marker=dict(
            color=["#2196f3", "#4caf50", "#ffc107"]
        ),
        textinfo="value+percent initial",
    ))
    fig.update_layout(title="Conversion Funnel Analysis")
    st.plotly_chart(fig, use_container_width=True)

def train_forecast_model(df: DataFrameType) -> Tuple[Optional[ModelType], Dict[str, float]]:
    """Train forecasting model with proper feature names."""
    if len(df) < 2:
        return None, {'error': 'Insufficient data for modeling'}

    try:
        X = pd.DataFrame({
            'leads': df['leads'],
            'appointments': df['appointments']
        })
        y = df['closings']

        model = LinearRegression()
        model.fit(X, y)

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

def predict_closings(model: Optional[ModelType], leads: float, appointments: float) -> Tuple[float, str]:
    """Make predictions with proper feature format."""
    if model is None:
        return 0.0, "No model available"

    try:
        input_data = pd.DataFrame({
            'leads': [leads],
            'appointments': [appointments]
        })

        prediction = model.predict(input_data)[0]
        return max(0.0, prediction), "success"
    except Exception as e:
        return 0.0, str(e)

def export_to_excel(df: DataFrameType, forecasts: Dict[str, Any]) -> bytes:
    """Create and return an Excel file with data and forecasts."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Historical Data', index=False)
        pd.DataFrame(forecasts).to_excel(writer, sheet_name='Forecasts', index=False)
    return output.getvalue()

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()

def main():
    """Main application function"""
    st.title("Sales Pipeline Analytics Dashboard")

    with st.container():
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Input & Upload", "📊 Analysis", "📈 Forecasting", "💾 Export"])

        with tab1:
            st.header("Input Your Data")
            col1, col2 = st.columns(2)
            
            with col1:
                num_leads = st.number_input("Number of Leads:", min_value=0, value=100)
                num_appointments = st.number_input("Number of Appointments:", min_value=0, value=50)

            with col2:
                num_closings = st.number_input("Number of Closings:", min_value=0, value=25)
                average_revenue_per_closing = st.number_input("Average Revenue per Closing ($):", 
                                                            min_value=0.0, value=10000.0)
                cost = st.number_input("Total Cost ($):", min_value=0.0, value=0.0)

            st.header("Upload Historical Data")
            
            # Sample data and templates
            st.subheader("Download Templates")
            col1, col2 = st.columns(2)
            
            with col1:
                sample_df = generate_sample_data()
                csv = sample_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Sample CSV",
                    data=csv,
                    file_name='sample_data.csv',
                    mime='text/csv'
                )
            
            with col2:
                excel_data = create_excel_template()
                st.download_button(
                    label="Download Excel Template",
                    data=excel_data,
                    file_name='template.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

            uploaded_file = st.file_uploader("Upload Your Data File", type=["csv", "xlsx"])

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)

                    data['month'] = pd.to_datetime(data['month'], errors='coerce')
                    data.dropna(subset=['month'], inplace=True)
                    st.session_state.historical_data = data
                    st.write("Uploaded Data Overview:")
                    st.dataframe(data)
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")

            # Add manual input to historical data
            new_row = pd.DataFrame({
                'month': [pd.Timestamp.now()],
                'leads': [num_leads],
                'appointments': [num_appointments],
                'closings': [num_closings],
                'cost': [cost]
            })

            if not st.session_state.historical_data.empty:
                st.session_state.historical_data = pd.concat(
                    [st.session_state.historical_data, new_row], 
                    ignore_index=True
                )

                st.write("Combined Data Overview:")
                st.dataframe(st.session_state.historical_data)
                create_metrics_dashboard(st.session_state.historical_data)

        with tab2:
            if not st.session_state.historical_data.empty:
                filtered_data = st.session_state.historical_data
                add_goals_tracking(filtered_data)
                create_metrics_dashboard(filtered_data)

                st.header("Performance Analysis")
                plot_interactive_trends(filtered_data)

                st.header("Conversion Analysis")
                plot_conversion_funnel(filtered_data)

                st.header("Seasonality Analysis")
                metric_choice = st.selectbox(
                    "Select Metric for Seasonality Analysis:", 
                    ['leads', 'appointments', 'closings']
                )

                if len(filtered_data) >= 24:
                    fig = plot_seasonality_analysis(filtered_data, metric_choice)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 24 months of data for reliable seasonality analysis.")
            else:
                st.warning("No data available for analysis. Please upload data.")

        with tab3:
            st.header("Forecast Analysis")
            st.subheader("Model Configuration")
            forecast_periods = st.slider("Forecast Periods (Months):", 1, 12, 3)

            if not st.session_state.historical_data.empty:
                # Train the forecasting model
                model, model_metrics = train_forecast_model(st.session_state.historical_data)

                if model is not None:
                    # Make predictions
                    forecasted_closings, prediction_status = predict_closings(
                        model, num_leads, num_appointments
                    )

                    if prediction_status == "success":
                        forecasted_revenue = forecasted_closings * average_revenue_per_closing

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Forecasted Closings",
                                f"{forecasted_closings:.1f}",
                                help="Predicted number of closings based on current leads and appointments"
                            )
                        with col2:
                            st.metric(
                                "Forecasted Revenue",
                                f"${forecasted_revenue:,.2f}",
                                help="Predicted revenue based on forecasted closings"
                            )

                        # Display model metrics
                        st.subheader("Model Performance Metrics")
                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Mean Absolute Error",
                            f"{model_metrics['mae']:.2f}",
                            help="Average absolute difference between predicted and actual values"
                        )
                        col2.metric(
                            "Root Mean Square Error",
                            f"{model_metrics['rmse']:.2f}",
                            help="Square root of the average squared differences"
                        )

                        if not np.isnan(model_metrics['r2']):
                            r2_value = model_metrics['r2']
                            col3.metric(
                                "R² Score",
                                f"{r2_value:.2f}",
                                help="Proportion of variance explained by the model"
                            )

                            if r2_value < 0.5:
                                st.warning("Warning: Model fit is poor. Predictions may be unreliable.")
                        else:
                            st.info("R² score requires more than two samples for calculation.")
                    else:
                        st.error(f"Prediction error: {prediction_status}")
            else:
                st.warning("No data available for forecasting. Please upload data.")

        with tab4:
            st.header("Export Data")
            if not st.session_state.historical_data.empty:
                # Prepare forecast data
                forecast_data = {
                    'Metric': ['Forecasted Closings', 'Forecasted Revenue'],
                    'Value': [forecasted_closings, forecasted_revenue],
                    'Date': [datetime.now().strftime('%Y-%m-%d')] * 2
                }

                # Create Excel file
                excel_data = export_to_excel(st.session_state.historical_data, forecast_data)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Full Report (Excel)",
                        data=excel_data,
                        file_name='sales_forecast_report.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        help="Download complete report including historical data and forecasts"
                    )

                with col2:
                    csv = st.session_state.historical_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Historical Data (CSV)",
                        data=csv,
                        file_name='historical_data.csv',
                        mime='text/csv',
                        help="Download historical data only"
                    )

                st.subheader("Export Preview")
                st.dataframe(st.session_state.historical_data, use_container_width=True)

                # Show summary metrics
                st.subheader("Summary Metrics")
                metrics_preview = calculate_metrics(st.session_state.historical_data)
                st.json(metrics_preview)
            else:
                st.warning("No data available for export. Please upload data.")

if __name__ == "__main__":
    main()
                
