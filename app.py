from flask import Flask, jsonify, request
from flask import request
import pandas as pd
import numpy as np
import joblib
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier
from flask_restful import reqparse

app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return "hey"
@app.route("/yes", methods=['GET'])
def yes():
    arg = request.args('arg1')
    return "hello world", arg1
@app.route('/predict', methods=['GET'])
def predict():
    #lr = xgb.XGBClassifier()
    params = {
    "learning_rate": 0.01,
    "max_depth": 3
    }
    lr = xgb.Booster(params)
    lr.load_model("model.json")
    age = 69.00
    cea = 1.020
    ibil = 9.800
    neu = 66.576
    menopause = 1.0
    ca125 = 113.300
    alb = 50.000
    he4 = 80.360
    glo = 31.700
    lym = 28.100
    if lr:
      json = request.get_json()  
      #model_columns = joblib.load("model_cols.pkl")
      age = 69.00
      cea = 1.020
      ibil = 9.800
      neu = 66.576
      menopause = 1.0
      ca125 = 113.300
      alb = 50.000
      he4 = 80.360
      glo = 31.700
      lym = 28.100

      x_test = {'Age': [age],
          'CEA': [cea],
          'IBIL': [ibil],
          'NEU': [neu],
          'Menopause': [menopause],        
          'CA125': [ca125],
          'ALB': [alb],
          'HE4': [he4],
          'GLO': [glo],              
          'LYM%': [lym],               
        }
      x_test = pd.DataFrame(x_test, columns= ['Age', 'CEA', 'IBIL', 'NEU', 'Menopause', 'CA125', 'ALB', 'HE4', 'GLO', 'LYM%'],dtype=float)
      #model_columns = joblib.load("model_cols.pkl")
      
      p = lr.predict(x_test)
      print("Cancer [0 - No Yes - 1] :\n Result : ",p[0])
      #vals=np.array(x_test)
      #prediction = lr.predict(x_test)
      #print("here:",prediction)        

      return jsonify({'prediction': str(p[0])})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='localhost', port=5000, debug=True)
