import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_filename = 'model.pkl'
model = joblib.load(model_filename)

# Streamlit app
st.title("Sales Prediction App")

# Input fields for user inputs (same as X_train features)
DayOfWeek = st.number_input("DayOfWeek (1 = Monday, 7 = Sunday)", min_value=1, max_value=7, value=1)
Promo = st.number_input("Promo (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
SchoolHoliday = st.number_input("SchoolHoliday (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
CompetitionDistance = st.number_input("CompetitionDistance", value=100.0)
CompetitionOpenSinceMonth = st.number_input("CompetitionOpenSinceMonth", min_value=1, max_value=12, value=1)
CompetitionOpenSinceYear = st.number_input("CompetitionOpenSinceYear", min_value=1900, max_value=2024, value=2020)
Promo2 = st.number_input("Promo2 (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
Promo2SinceWeek = st.number_input("Promo2SinceWeek", min_value=1, max_value=52, value=1)
Promo2SinceYear = st.number_input("Promo2SinceYear", min_value=1900, max_value=2024, value=2020)
Day = st.number_input("Day of Month", min_value=1, max_value=31, value=1)
Month = st.number_input("Month", min_value=1, max_value=12, value=1)
Year = st.number_input("Year", min_value=1900, max_value=2024, value=2020)

# Input for StoreType and Assortment
store_type_options = ['StoreType_2', 'StoreType_3', 'StoreType_4', 'StoreType_5']
assortment_options = ['Assortment_2', 'Assortment_3', 'Assortment_4']

# StoreType input
selected_store_type = st.selectbox("StoreType", options=store_type_options)
store_type_values = {key: 0 for key in store_type_options}
store_type_values[selected_store_type] = 1

# Assortment input
selected_assortment = st.selectbox("Assortment", options=assortment_options)
assortment_values = {key: 0 for key in assortment_options}
assortment_values[selected_assortment] = 1

# Create a DataFrame for input data
input_data = {
    'DayOfWeek': [DayOfWeek],
    'Promo': [Promo],
    'SchoolHoliday': [SchoolHoliday],
    'CompetitionDistance': [CompetitionDistance],
    'CompetitionOpenSinceMonth': [CompetitionOpenSinceMonth],
    'CompetitionOpenSinceYear': [CompetitionOpenSinceYear],
    'Promo2': [Promo2],
    'Promo2SinceWeek': [Promo2SinceWeek],
    'Promo2SinceYear': [Promo2SinceYear],
    'Day': [Day],
    'Month': [Month],
    'Year': [Year],
    **store_type_values,
    **assortment_values
}

input_df = pd.DataFrame(input_data)

# Create a new StandardScaler instance and fit it on the relevant features
scaler = StandardScaler()

# Fit the scaler on the specific features used for scaling
scaler.fit(input_df[['DayOfWeek', 'Promo', 'SchoolHoliday', 
                      'CompetitionDistance', 'CompetitionOpenSinceMonth', 
                      'CompetitionOpenSinceYear', 'Promo2', 
                      'Promo2SinceWeek', 'Promo2SinceYear']])

# Scale the input data using the newly fitted scaler
input_df_scaled = scaler.transform(input_df[['DayOfWeek', 'Promo', 'SchoolHoliday', 
    'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
    'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear']])

# Combine scaled features with the remaining features (StoreType and Assortment dummies, plus Day, Month, Year)
input_df_final = np.hstack((input_df_scaled, 
                             input_df[['Day', 'Month', 'Year']].values,
                             input_df.loc[:, store_type_options + assortment_options].values))

# Define mean and std dev for sales reverse transformation
mean = 8.757564
std_dev = 0.425278

# Predict button
if st.button("Predict"):
    # Make the prediction using the model
    log_sales_prediction = model.predict(input_df_final)

    # Reverse scaling to get the prediction in the log scale
    sales_prediction_scaled = (log_sales_prediction * std_dev) + mean

    # Reverse log transformation to get the final sales prediction
    sales_prediction_original = np.exp(sales_prediction_scaled)

    # Display the prediction
    st.success(f"The predicted sales are: {sales_prediction_original[0]:.2f}")
