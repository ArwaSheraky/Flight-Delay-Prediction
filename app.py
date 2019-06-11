from flask import Flask, render_template, request
from sklearn.externals import joblib
import os
from datetime import datetime

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict_flight_delay', methods=['POST', 'GET'])
def predict_flight_delay():
    
    # get the parameters
    origin_airport = str(request.form['origin_airport'])
    dest_airport = str(request.form['dest_airport'])
    distance = float(request.form['distance'])
    sched_arr = float(request.form['distance'])
    sched_dep = float(request.form['distance'])
    date = datetime(request.form['date'])

    # load the model and predict
    model = joblib.load('model/regression.pkl')
    prediction = model.predict([[origin_airport, dest_airport, distance, sched_arr, sched_dep, date]])
    predicted_delay = prediction.round(1)[0]

    return render_template('results.html',
                           origin_airport= str(origin_airport),
                           dest_airport= str(dest_airport),
                           distance= float(distance),
                           sched_arr= float(sched_arr),
                           sched_dep= float(sched_dep),
                           date = datetime(date),
                           predicted_delay="{:,}".format(predicted_delay)
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
