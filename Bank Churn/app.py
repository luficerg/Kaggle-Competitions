from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            CustomerId = int(request.form['CustomerId'])
            Surname = request.form['Surname']
            CreditScore = int(request.form['CreditScore'])
            Geography = request.form['Geography']
            Gender = request.form['Gender']
            Age = float(request.form['Age'])
            Tenure = int(request.form['Tenure'])
            Balance = float(request.form['Balance'])
            NumOfProducts = int(request.form['NumOfProducts'])
            HasCrCard = float(request.form['HasCrCard'])
            IsActiveMember = float(request.form['IsActiveMember'])
            EstimatedSalary = float(request.form['EstimatedSalary'])

            data = [CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
            column_names = ['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

            df = pd.DataFrame([data], columns=column_names)

            
            obj = PredictionPipeline()
            predict = obj.predict(df)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)