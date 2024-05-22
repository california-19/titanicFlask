from flask import Flask, render_template, request
import numpy as np
import xgboost as xgb
import json

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello_world():

    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html')
    else:
        input_data = request.form['values']
        pred = make_prediction(input_data)
        return render_template('index.html', prediction=pred, input_data=input_data)
    
def make_prediction(input_data):
    # input_data = np.array([[2,30,1,2,40,0,1],[3,30,1,2,40,1,0]])
    # input_data = np.reshape(input_data, (-1, 7))
    input_data = np.array([int(x) for x in input_data.split(',')]).reshape(-1, 7)
    print(input_data, input_data.shape)
    # input_data = np.array(input_data).reshape(7,1)
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model('../saved_models/titanic_model.json')
    model_output = loaded_model.predict(input_data)
    model_output = str(model_output)

    return model_output