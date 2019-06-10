from flask import Flask, render_template, request
from sklearn.externals import joblib
import os

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict_flight_delay', methods=['POST', 'GET'])
def predict_flight_delay():
    
    # get the parameters
    bedrooms = float(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])
    sqft_living15 = float(request.form['sqft_living15'])
    grade = float(request.form['grade'])
    condition = float(request.form['condition'])

    # load the model and predict
    model = joblib.load('model/linear_regression.pkl')
    prediction = model.predict([[bedrooms, bathrooms, sqft_living15, grade, condition]])
    predicted_price = prediction.round(1)[0]

    return render_template('results.html',
                           bedrooms=int(bedrooms),
                           bathrooms=int(bathrooms),
                           sqft_living15=int(sqft_living15),
                           grade=int(grade),
                           condition=int(condition),
                           predicted_price="{:,}".format(predicted_price)
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
