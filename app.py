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
    airline = str(request.form['airline'])
    distance = float(request.form['distance'])
    sched_arr = str(request.form['sched_arr'])
    sched_dep = str(request.form['sched_dep'])
    date = datetime.strptime(request.form['date'], '%Y-%m-%d')
    date = date.date()

    # prepare the data
    # X_columns = ['SCHEDULED_DEPARTURE','DATE','	SCHEDULED_ARRIVAL','AIRLINE_NAME_Southwest Airlines Co.',
    # 'AIRLINE_NAME_Delta Air Lines Inc.','AIRLINE_NAME_Spirit Air Lines']

    # # load the model and predict
    # model = joblib.load('gb_model.pkl')
    # prediction = model.predict([[origin_airport, dest_airport, distance, sched_arr, sched_dep, date, airline]])
    # predicted_delay = prediction.round(1)[0]

    return render_template('results.html',
                           origin_airport= (origin_airport),
                           dest_airport= (dest_airport),
                           airline = (airline),
                           distance= (distance),
                           sched_arr= (sched_arr),
                           sched_dep= (sched_dep),
                           date = (date)#,
                           #predicted_delay="{:,}".format(predicted_delay)
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
