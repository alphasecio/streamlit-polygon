import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go
import calendar
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

def calculate_metrics(df):
    """Calculate business metrics including cost metrics."""
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
        'Best Month (Closings)': df.loc[df['closings'].idxmax(), 'month'].strftime('%b %y'),
        'Worst Month (Closings)': df.loc[df['closings'].idxmin(), 'month'].strftime('%b %y'),
    }
    return metrics

def create_metrics_dashboard(df):
    """Create metrics dashboard for display."""
    st.header("Key Metrics Dashboard")
    metrics = calculate_metrics(df)
    col1, col2, col3 = st.columns(3)
    for i, (metric, value) in enumerate(metrics.items()):
        with [col1, col2, col3][i % 3]:
            if isinstance(value, float):
                st.metric(metric, f"{value:.2f}")
            else:
                st.metric(metric, value)

def export_to_excel(df, forecasts):
    """Create and return an Excel file with data and forecasts."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Historical data sheet
        df.to_excel(writer, sheet_name='historical data', index=False)

        # Forecasts: put forecasts in a DataFrame format
        forecast_df = pd.DataFrame(forecasts)
        forecast_df.to_excel(writer, sheet_name='forecasts', index=False)

    return output.getvalue()

# Primary app logic
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()

with st.container():
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Input & Upload", "Analysis", "Forecasting", "Export"])

    with tab1:
        # Manual input section
        st.header("Input Your Data Manually")
        col1, col2 = st.columns(2)

        with col1:
            num_leads = st.number_input("Number of Leads:", min_value=0, value=100)
            num_appointments = st.number_input("Number of Appointments:", min_value=0, value=50)

        with col2:
            num_closings = st.number_input("Number of Closings:", min_value=0, value=25)
            average_revenue_per_closing = st.number_input("Average Revenue Per Closing ($):", 
                                                            min_value=0.0, 
                                                            value=10000.0)
            cost = st.number_input("Total Cost:", min_value=0.0, value=5000.0)  # New input for total cost

        # File upload section
        st.header("Upload Historical Data")
        uploaded_file = st.file_uploader("Upload Your Data File", type=["csv", "xlsx"])

        if uploaded_file is not None:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                st.session_state.historical_data = pd.read_csv(uploaded_file)
            else:
                st.session_state.historical_data = pd.read_excel(uploaded_file)

            st.session_state.historical_data['month'] = pd.to_datetime(st.session_state.historical_data['month'])
            st.write("Uploaded Data Overview:")
            st.dataframe(st.session_state.historical_data)

        # Combine manual input with uploaded data
        if not st.session_state.historical_data.empty:  # If there is uploaded data
            # Create a new row with the manual input data
            new_row = pd.DataFrame({
                'month': [pd.Timestamp('now')],
                'leads': [num_leads],
                'appointments': [num_appointments],
                'closings': [num_closings],
                'cost': [cost]  # Include cost in the new row
            })
            # Append the manual input data to the historical data DataFrame
            st.session_state.historical_data = pd.concat([st.session_state.historical_data, new_row], ignore_index=True)

        else:  # No uploaded data, create a DataFrame with manual input only
            st.session_state.historical_data = pd.DataFrame({
                'month': [pd.Timestamp('now')],
                'leads': [num_leads],
                'appointments': [num_appointments],
                'closings': [num_closings],
                'cost': [cost]  # Include cost in the DataFrame
            })

        # Display the combined data overview
        st.write("Combined Data Overview:")
        st.dataframe(st.session_state.historical_data)

        # Always recalculate metrics based on the current data
        create_metrics_dashboard(st.session_state.historical_data)

    with tab2:
        # Perform additional analysis if data is available
        if not st.session_state.historical_data.empty:
            st.sidebar.header("Analysis Filters")
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(
                    st.session_state.historical_data['month'].min(),
                    st.session_state.historical_data['month'].max()
                )
            )

            # Filter data based on the selected date range
            mask = (st.session_state.historical_data['month'].dt.date >= date_range[0]) & \
                   (st.session_state.historical_data['month'].dt.date <= date_range[1])
            filtered_data = st.session_state.historical_data.loc[mask]

            # Goals tracking
            add_goals_tracking(filtered_data)

            # Recalculate and display metrics from filtered data
            create_metrics_dashboard(filtered_data)

            # Detailed performance metrics
            st.header("Detailed Performance Metrics")
            metrics = calculate_metrics(filtered_data)
            col1, col2, col3 = st.columns(3)
            for i, (metric, value) in enumerate(metrics.items()):
                with [col1, col2, col3][i % 3]:
                    if isinstance(value, float):
                        st.metric(metric, f"{value:.2f}")
                    else:
                        st.metric(metric, value)

            # Seasonality analysis
            st.header("Seasonality Analysis")
            metric_choice = st.selectbox(
                "Select Metric for Seasonality Analysis:",
                ['leads', 'appointments', 'closings']
            )
            if len(filtered_data) >= 12:  # Need at least 12 months for seasonality
                fig = plot_seasonality_analysis(filtered_data, metric_choice)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 12 months of data for seasonality analysis.")

            # Historical trends
            st.header("Historical Trends")
            plot_interactive_trends(filtered_data)

            # Conversion funnel
            st.header("Conversion Funnel")
            plot_conversion_funnel(filtered_data)
        else:
            st.warning("No data available for analysis. Please upload data or enter manual inputs.")

    with tab3:
        # Forecast section
        st.header("Forecast Analysis")

        # Model settings
        st.subheader("Model Configuration")
        forecast_periods = st.slider("Forecast Periods (Months):", 1, 12, 3)

        if not st.session_state.historical_data.empty:
            # Fit model and make predictions
            x = st.session_state.historical_data[['leads', 'appointments']]
            y = st.session_state.historical_data['closings']
            model = LinearRegression()
            model.fit(x, y)

            # Make current prediction
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

            # Model metrics
            st.subheader("Model Performance Metrics")
            y_pred = model.predict(x)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = model.score(x, y)

            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error", f"{mae:.2f}")
            col2.metric("Root Mean Square Error", f"{rmse:.2f}")
            col3.metric("R² Score", f"{r2:.2f}")

            if r2 < 0.5:
                st.warning("Warning: Model fit is poor. Predictions may be unreliable.")
        else:
            st.warning("No data available for forecasting. Please upload data or enter manual inputs.")

    with tab4:
        # Export section
        st.header("Export Data")

        # Prepare forecast data for export
        if not st.session_state.historical_data.empty:
            forecast_data = {
                'Metric': ['Forecasted Closings', 'Forecasted Revenue'],
                'Value': [forecasted_closings, forecasted_revenue]
            }

            # Create Excel file
            excel_data = export_to_excel(filtered_data, forecast_data)

            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Full Report (Excel)",
                    data=excel_data,
                    file_name='sales_forecast_report.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

            with col2:
                # CSV export
                csv = st.session_state.historical_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Historical Data (CSV)",
                    data=csv,
                    file_name='historical_data.csv',
                    mime='text/csv'
                )

            # Display export preview
            st.subheader("Export Preview")
            st.dataframe(st.session_state.historical_data)
        else:
            st.warning("No data available for export. Please upload data or enter manual inputs.")
