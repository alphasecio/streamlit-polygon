import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title("Lead, Appointment, and Closing Stats Forecaster")

# Input section for user manual input (optional)
st.header("Input Your Data Manually (Optional)")
num_leads = st.number_input("Number of Leads:", min_value=0)
num_appointments = st.number_input("Number of Appointments:", min_value=0)
num_closings = st.number_input("Number of Closings:", min_value=0)

# Define average revenue per closing (you can set this to your expected value)
average_revenue_per_closing = st.number_input("Average Revenue per Closing ($):", min_value=0.0, value=10000.0)

# Create a CSV file with headers and sample data for download
sample_data = {
    'month': ['2023-01-01', '2023-02-01', '2023-03-01'],
    'leads': [0, 0, 0],
    'appointments': [0, 0, 0],
    'closings': [10000, 20000, 30000]  # Sample dollar amounts for closings
}

# Create a DataFrame
sample_df = pd.DataFrame(sample_data)

# Save DataFrame to a CSV file in memory
csv_file = sample_df.to_csv(index=False).encode('utf-8')

# Create a download button for the CSV file
st.download_button(
    label="Download Example CSV",
    data=csv_file,
    file_name='example_data.csv',
    mime='text/csv',
)

# Prompt for CSV file structure
st.info("Please upload a CSV file with the following columns: 'month', 'leads', 'appointments', 'closings'.")
st.write("**Example CSV Format:**")
st.write("```\nmonth,leads,appointments,closings\n2023-01-01,0,0,10000\n2023-02-01,0,0,20000\n```")

uploaded_file = st.file_uploader("Upload Your Updated CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        historical_data = pd.read_csv(uploaded_file)

        # Validate the structure of the uploaded data
        required_columns = ['month', 'leads', 'appointments', 'closings']
        if not all(column in historical_data.columns for column in required_columns):
            st.error("Uploaded file must contain the following columns: " + ", ".join(required_columns))
        else:
            # Check for non-negative values
            if (historical_data[['leads', 'appointments', 'closings']] < 0).any().any():
                st.error("Values in 'leads', 'appointments', and 'closings' must be non-negative.")
            else:
                # Handle missing values using SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                historical_data[['leads', 'appointments']] = imputer.fit_transform(historical_data[['leads', 'appointments']])

                # Proceed with the analysis if data is valid
                historical_data['month'] = pd.to_datetime(historical_data['month'])  # Ensure month is in datetime format
                historical_data.sort_values('month', inplace=True)  # Sort by month

                # Fit the model
                X = historical_data[['leads', 'appointments']]
                y = historical_data['closings']
                model = LinearRegression()
                model.fit(X, y)

                # Make a forecast based on the user input
                input_data = np.array([[num_leads, num_appointments]]).reshape(1, -1)
                forecasted_closings = model.predict(input_data)[0]

                # Calculate forecasted revenue based on closings
                forecasted_revenue = forecasted_closings * average_revenue_per_closing 

                # Button to calculate forecast
                if st.button("Forecast"):
                    st.success(f"Forecasted Revenue from Closings: ${forecasted_revenue:.2f}")

                    # Visualizing the data
                    st.header("Forecast Visualization")
                    forecast_data = {
                        'Categories': ['Leads', 'Appointments', 'Forecasted Closings'],
                        'Values': [num_leads, num_appointments, forecasted_closings]
                    }
                    df_forecast = pd.DataFrame(forecast_data)
                    st.bar_chart(df_forecast.set_index('Categories'), use_container_width=True)

                    # Line chart for historical trends
                    st.header("Historical Data Trends")
                    plt.figure(figsize=(10, 5))
                    sns.lineplot(data=historical_data, x='month', y='leads', label='Leads', color='blue')
                    sns.lineplot(data=historical_data, x='month', y='appointments', label='Appointments', color='orange')
                    sns.lineplot(data=historical_data, x='month', y='closings', label='Closings', color='green')
                    plt.title('Historical Trends')
                    plt.xlabel('Month')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45)
                    plt.legend()
                    st.pyplot(plt)

                    # Displaying the uploaded historical data alongside the forecast
                    st.header("Data Overview")
                    st.dataframe(historical_data)

                    # Displaying forecasted revenue in context
                    st.write("**Forecasted Revenue Based on User Input:**")
                    st.write(f"Forecasted Revenue: ${forecasted_revenue:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
