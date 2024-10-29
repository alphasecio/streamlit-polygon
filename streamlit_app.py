import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# title of the app
st.title("Lead, Appointment, and Closing Stats Forecaster")

# input section for user manual input (optional)
st.header("Input Your Data Manually (Optional)")
num_leads = st.number_input("Number of Leads:", min_value=0)
num_appointments = st.number_input("Number of Appointments:", min_value=0)
num_closings = st.number_input("Number of Closings:", min_value=0)

# define average revenue per closing (you can set this to your expected value)
average_revenue_per_closing = st.number_input("Average Revenue Per Closing ($):", min_value=0.0, value=10000.0)

# create a csv file with headers and sample data for download
sample_data = {
    'month': ['2023-01-01', '2023-02-01', '2023-03-01'],
    'leads': [0, 0, 0],
    'appointments': [0, 0, 0],
    'closings': [10000, 20000, 30000]  # sample dollar amounts for closings
}

# create a dataframe
sample_df = pd.DataFrame(sample_data)

# save dataframe to a csv file in memory
csv_file = sample_df.to_csv(index=False).encode('utf-8')

# create a download button for the csv file
st.download_button(
    label="Download Example CSV",
    data=csv_file,
    file_name='example_data.csv',
    mime='text/csv',
)

# prompt for csv file structure
st.info("Please upload a CSV file with the following columns: 'month', 'leads', 'appointments', 'closings'.")
st.write("**Example CSV Format:**")
st.write("```\nmonth,leads,appointments,closings\n2023-01-01,0,0,10000\n2023-02-01,0,0,20000\n```")

uploaded_file = st.file_uploader("Upload Your Updated CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # read the uploaded csv file
        historical_data = pd.read_csv(uploaded_file)

        # validate the structure of the uploaded data
        required_columns = ['month', 'leads', 'appointments', 'closings']
        if not all(column in historical_data.columns for column in required_columns):
            st.error("Uploaded file must contain the following columns: " + ", ".join(required_columns))
        else:
            # check for non-negative values
            if (historical_data[['leads', 'appointments', 'closings']] < 0).any().any():
                st.error("Values in 'leads', 'appointments', and 'closings' must be non-negative.")
            else:
                # handle missing values using SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                historical_data[['leads', 'appointments']] = imputer.fit_transform(historical_data[['leads', 'appointments']])

                # proceed with the analysis if data is valid
                historical_data['month'] = pd.to_datetime(historical_data['month'])  # ensure month is in datetime format
                historical_data.sort_values('month', inplace=True)  # sort by month

                # fit the model
                x = historical_data[['leads', 'appointments']]
                y = historical_data['closings']
                model = LinearRegression()
                model.fit(x, y)

                # make a forecast based on the user input
                input_data = np.array([[num_leads, num_appointments]]).reshape(1, -1)
                forecasted_closings = model.predict(input_data)[0]

                # calculate forecasted revenue based on closings
                forecasted_revenue = forecasted_closings * average_revenue_per_closing 

                # button to calculate forecast
                if st.button("Forecast"):
                    st.success(f"Forecasted revenue from closings: ${forecasted_revenue:.2f}")

                    # visualizing the data
                    st.header("Forecast Visualization")
                    forecast_data = {
                        'Categories': ['Leads', 'Appointments', 'Forecasted Closings'],
                        'Values': [num_leads, num_appointments, forecasted_closings]
                    }
                    df_forecast = pd.DataFrame(forecast_data)
                    st.bar_chart(df_forecast.set_index('Categories'), use_container_width=True)

                    # line chart for historical trends
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

                    # displaying the uploaded historical data alongside the forecast
                    st.header("Data Overview")
                    st.dataframe(historical_data)

                    # displaying forecasted revenue in context
                    st.write("**Forecasted Revenue Based on User Input:**")
                    st.write(f"Forecasted revenue: ${forecasted_revenue:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
