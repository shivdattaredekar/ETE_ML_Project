import os 
from flask import Flask, render_template, request
import numpy as np
import joblib
from Mlproject.pipeline.prediction import PredictionPipeline

app = Flask(__name__) # initializing a flask app

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


# Route to display the home page with the form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Collecting input values from the form
            LIMIT_BAL = float(request.form['LIMIT_BAL'])
            SEX = int(request.form['SEX'])
            EDUCATION = int(request.form['EDUCATION'])
            MARRIAGE = int(request.form['MARRIAGE'])
            AGE = int(request.form['AGE'])
            PAY_0 = int(request.form['PAY_0'])
            PAY_2 = int(request.form['PAY_2'])
            PAY_3 = int(request.form['PAY_3'])
            PAY_4 = int(request.form['PAY_4'])
            PAY_5 = int(request.form['PAY_5'])
            PAY_6 = int(request.form['PAY_6'])
            BILL_AMT1 = float(request.form['BILL_AMT1'])
            BILL_AMT2 = float(request.form['BILL_AMT2'])
            BILL_AMT3 = float(request.form['BILL_AMT3'])
            BILL_AMT4 = float(request.form['BILL_AMT4'])
            BILL_AMT5 = float(request.form['BILL_AMT5'])
            BILL_AMT6 = float(request.form['BILL_AMT6'])
            PAY_AMT1 = float(request.form['PAY_AMT1'])
            PAY_AMT2 = float(request.form['PAY_AMT2'])
            PAY_AMT3 = float(request.form['PAY_AMT3'])
            PAY_AMT4 = float(request.form['PAY_AMT4'])
            PAY_AMT5 = float(request.form['PAY_AMT5'])
            PAY_AMT6 = float(request.form['PAY_AMT6'])
            
            # Creating the data array with the correct types
            data = [LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, 
                    PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4,
                    BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]
            
            data = np.array(data).reshape(1, 23)
            
            # Loading the pre-fitted scaler and transforming the data
            scaler = joblib.load('artifacts/model_training/scaler.joblib')
            data = scaler.transform(data)
            
            # Making the prediction
            prediction_pipeline = PredictionPipeline()
            result = prediction_pipeline.predict(data)
            
            # Interpreting the result
            response = 'The client is not likely to default.' if result == 0 else 'The client is likely to default.'
            
            return render_template('index.html', prediction=response)
        
        except Exception as e:
            raise e
    
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)