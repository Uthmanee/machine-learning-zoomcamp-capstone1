import pickle
from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb


model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('depression_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    dX = xgb.DMatrix(X)

    y_pred = model.predict(dX)[0]
    depression_prediction = y_pred >= 0.5

    result = {
        'depression_probability': float(y_pred),
        'depression_prediction': bool(depression_prediction)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

# Instead of running a Flask app with app.run(), you would run it in production like this: gunicorn predict:app
