from flask import Flask, jsonify, request
from flask import request
import pandas as pd
import numpy as np
import joblib
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier
from flask_restful import reqparse
from sklearn.metrics import f1_score

app = Flask(__name__)


@app.route("/", methods=['GET'])
def hello():
    return "hey"
    
@app.route('/test', methods=['GET'])
def test():
    train = pd.read_csv("OvarianCancer.csv")
    X = train.copy()
    y = X.pop('TYPE')

    size = int(X.shape[0] * .8)
    train = X[:size]
    y_train = y[:size]
    test = X[size:]
    y_test = y[size:]

    params = {
    'n_estimators': 100,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'max_depth': 7,
    'min_child_weight': 1,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'eta': 0.2
    }

    dtrain = xgb.DMatrix(train, y_train)
    dtest = xgb.DMatrix(test)

    model = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,
    nfold=5,
    early_stopping_rounds=100
    )

    # Fit
    final_gb = xgb.train(params, dtrain, num_boost_round=len(model))

    preds = final_gb.predict(dtest)
    f1 = f1_score(y_test, preds)
    print(f1)
    
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
