pip install plotly


import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

# Loading my data
@st.cache
def load_data():
    # Load the CSV file
    df = pd.read_csv("df_combined_sorted_nonzero.csv")
    # Ensure the 'Date' column is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Loading the data into a DataFrame
df_combined_sorted_nonzero = load_data()

st.title("Farm Meat Production - Time Series Analysis")

# Adding the sidebar filters
st.sidebar.header("Filters")
country = st.sidebar.selectbox("Select Country", df_combined_sorted_nonzero['Country'].unique())
meat_type = st.sidebar.selectbox("Select Meat Type", df_combined_sorted_nonzero['Meat Type'].unique())
value_category = st.sidebar.multiselect(
    "Select Value Category (optional)", 
    df_combined_sorted_nonzero['Value Category'].unique(),
    default=df_combined_sorted_nonzero['Value Category'].unique()
)

# Filtering the data
df_filtered = df_combined_sorted_nonzero[
    (df_combined_sorted_nonzero['Country'] == country) &
    (df_combined_sorted_nonzero['Meat Type'] == meat_type) &
    (df_combined_sorted_nonzero['Value Category'].isin(value_category))
]

# Ensuring that the data is sorted
df_filtered = df_filtered.sort_values(by="Date")

# Adding the time Series Line Plot
st.subheader(f"Time Series: {meat_type} in {country}")
if df_filtered.empty:
    st.write("No data available for the selected filters.")
else:
    fig = px.line(
        df_filtered, 
        x="Date", 
        y="Value", 
        title=f"Time Series of {meat_type} in {country}",
        labels={"Value": "Meat Production (tonnes)", "Date": "Year"},
        template="plotly_white"
    )
    st.plotly_chart(fig)

# Showing the seasonal decomposition
st.subheader("Seasonal Decomposition")
if len(df_filtered) >= 24:  # Require at least 2 years of monthly data
    # Set 'Date' as index for decomposition
    df_filtered.set_index('Date', inplace=True)
    df_filtered.sort_index(inplace=True)

    # Performing seasonal decomposition
    decomposition = seasonal_decompose(df_filtered['Value'], model='additive', period=12)

    # Plotting seasonal decomposition
    st.write("Observed Trend")
    st.line_chart(decomposition.observed)
    st.write("Trend")
    st.line_chart(decomposition.trend)
    st.write("Seasonality")
    st.line_chart(decomposition.seasonal)
    st.write("Residuals")
    st.line_chart(decomposition.resid)
else:
    st.write("Not enough data for seasonal decomposition. At least 24 monthly observations are required.")
