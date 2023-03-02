import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Classifying House Prices in California')
st.markdown('Toy model to play to classify the price that a house in California should have with certain characteristics')

st.header('House Features')
col1, col2 = st.columns(2)
with col1:
    households = st.number_input('Households: ', min_value=1)
    m_age = st.number_input('Housing Median Age: ', min_value=0)
    bedrooms = st.number_input('Total bedrooms: ', min_value=1)
    rooms = st.number_input('Total rooms: ', min_value=1)

with col2:
    latitude = st.number_input('Latitude')
    longitude = st.number_input('Longitude', max_value=0)
    population = st.number_input('Population', min_value=0)
    income = st.number_input('Median income', min_value=0)

ocean_proximity = st.selectbox("Ocean proximity: ", ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

if st.button('Estimate Price'):
    if ocean_proximity == '<1H OCEAN':
        op = 0
    elif ocean_proximity == 'INLAND':
        op = 1
    elif ocean_proximity == 'ISLAND':
        op = 2
    elif ocean_proximity == 'NEAR BAY':
        op = 3
    elif ocean_proximity == 'NEAR OCEAN':
        op = 4
    data = pd.DataFrame({
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [m_age],
            'total_rooms': [rooms],
            'total_bedrooms': [bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [income],
            'ocean_proximity': [ocean_proximity]
        })
    result = predict(data)
    st.text('$' + str(result[0]))
    st.snow()