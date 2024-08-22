import os 
import numpy as np
import pandas as pd
from Mlproject.pipeline.prediction import PredictionPipeline
import streamlit as st


st.title('Welcome to default prediction')

st.sidebar.header('Please enter your values here')

# Example input in the sidebar (you can replace these with the actual inputs you need)
LIMIT_BAL = st.sidebar.slider('LIMIT_BAL:', min_value=0)
SEX = st.sidebar.slider('SEX:', min_value=0, max_value=100, value=50)
EDUCATION = st.sidebar.slider('EDUCATION:', min_value=0, max_value=100, value=50)
MARRIAGE = st.sidebar.slider('Input 1:', min_value=0, max_value=100, value=50)
AGE = st.sidebar.slider('MARRIAGE:', min_value=0, max_value=100, value=50)
PAY_0 = st.sidebar.slider('PAY_0:', min_value=0, max_value=100, value=50)
PAY_2 = st.sidebar.slider('PAY_2:', min_value=0, max_value=100, value=50)
PAY_3 = st.sidebar.slider('PAY_3:', min_value=0, max_value=100, value=50)
PAY_4 = st.sidebar.slider('PAY_4:', min_value=0, max_value=100, value=50)
PAY_5 = st.sidebar.slider('PAY_5:', min_value=0, max_value=100, value=50)
PAY_6 = st.sidebar.slider('PAY_6:', min_value=0, max_value=100, value=50)
BILL_AMT1 = st.sidebar.slider('BILL_AMT1:', min_value=0, max_value=100, value=50)
BILL_AMT2 = st.sidebar.slider('BILL_AMT2:', min_value=0, max_value=100, value=50)
BILL_AMT3 = st.sidebar.slider('BILL_AMT3:', min_value=0, max_value=100, value=50)
BILL_AMT4 = st.sidebar.slider('BILL_AMT4:', min_value=0, max_value=100, value=50)
BILL_AMT5 = st.sidebar.slider('BILL_AMT5:', min_value=0, max_value=100, value=50)
BILL_AMT6 = st.sidebar.slider('BILL_AMT6:', min_value=0, max_value=100, value=50)
PAY_AMT1 = st.sidebar.slider('PAY_AMT1:', min_value=0, max_value=100, value=50)
PAY_AMT2 = st.sidebar.slider('PAY_AMT2:', min_value=0, max_value=100, value=50)
PAY_AMT3 = st.sidebar.slider('PAY_AMT3:', min_value=0, max_value=100, value=50)
PAY_AMT4 = st.sidebar.slider('PAY_AMT4:', min_value=0, max_value=100, value=50)
PAY_AMT5 = st.sidebar.slider('PAY_AMT5:', min_value=0, max_value=100, value=50)
PAY_AMT6 = st.sidebar.slider('PAY_AMT6:', min_value=0, max_value=100, value=50)


# Use the inputs in your prediction pipeline, if applicable
if st.sidebar.button('Predict'):
    try :
        data = [LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,	
            PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,
                BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6]
        
        data = np.array(data).reshape(1, 23)
        prediction_pipeline = PredictionPipeline()
        result = prediction_pipeline.predict(data)
        st.write(f'Prediction result goes here: {result}')
    except Exception as e:
        print('The Exception message is: ',e)

