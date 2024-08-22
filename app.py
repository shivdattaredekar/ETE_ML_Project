
# Importing necessary libraries for the project
import numpy as np
from Mlproject.pipeline.prediction import PredictionPipeline
import streamlit as st
import joblib

st.title('Welcome to Default Prediction')

st.sidebar.header('Please enter your values here')

# Collecting all slider inputs with appropriate types
SEX = int(st.sidebar.slider('SEX:', min_value=1, max_value=2, value=1))
EDUCATION = int(st.sidebar.slider('EDUCATION:', min_value=0, max_value=6, value=6))
MARRIAGE = int(st.sidebar.slider('MARRIAGE:', min_value=0, max_value=3, value=1))
AGE = int(st.sidebar.slider('AGE:', min_value=18, max_value=100, value=30))
PAY_0 = int(st.sidebar.slider('PAY_0:', min_value=-2, max_value=8, value=0))
PAY_2 = int(st.sidebar.slider('PAY_2:', min_value=-2, max_value=8, value=0))
PAY_3 = int(st.sidebar.slider('PAY_3:', min_value=-2, max_value=8, value=0))
PAY_4 = int(st.sidebar.slider('PAY_4:', min_value=-2, max_value=8, value=0))
PAY_5 = int(st.sidebar.slider('PAY_5:', min_value=-2, max_value=8, value=0))
PAY_6 = int(st.sidebar.slider('PAY_6:', min_value=-2, max_value=8, value=0))
LIMIT_BAL = float(st.sidebar.number_input(label = 'LIMIT_BAL'))
BILL_AMT1 = float(st.sidebar.number_input(label = 'BILL_AMT1'))
BILL_AMT2 = float(st.sidebar.number_input(label = 'BILL_AMT2'))
BILL_AMT3 = float(st.sidebar.number_input(label = 'BILL_AMT3'))
BILL_AMT4 = float(st.sidebar.number_input(label = 'BILL_AMT4'))
BILL_AMT5 = float(st.sidebar.number_input(label = 'BILL_AMT5'))
BILL_AMT6 = float(st.sidebar.number_input(label = 'BILL_AMT6'))
PAY_AMT1 = float(st.sidebar.number_input(label = 'PAY_AMT1'))
PAY_AMT2 = float(st.sidebar.number_input(label = 'PAY_AMT2'))
PAY_AMT3 = float(st.sidebar.number_input(label = 'PAY_AMT3'))
PAY_AMT4 = float(st.sidebar.number_input(label = 'PAY_AMT4'))
PAY_AMT5 = float(st.sidebar.number_input(label = 'PAY_AMT5'))
PAY_AMT6 = float(st.sidebar.number_input(label = 'PAY_AMT6'))

# Use the inputs in your prediction pipeline, if applicable
if st.sidebar.button('Predict'):
    try:
        # Creating the data array with the correct types
        data = [LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, 
                PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4,
                BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]
        
        data = np.array(data).reshape(1, 23)
        
        # Assuming the scaler is pre-fitted, load it here
        scaler = joblib.load('artifacts\\model_training\\scaler.joblib')
        data = scaler.transform(data)
        
        prediction_pipeline = PredictionPipeline()
        result = prediction_pipeline.predict(data)
        response = '' 
        if result == 1:
            response = 'The client is likely to default.'
        else :
            response = 'The client is not likely to default.'
         


        st.write(f'Prediction result: {response}')
    except Exception as e:
        st.write(f'An error occurred: {e}')
        
        
