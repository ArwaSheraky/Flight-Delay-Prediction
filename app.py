from flask import Flask, render_template, request
from sklearn.externals import joblib
import os
from datetime import datetime
import pandas as pd
import _pickle as cPickle

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
    date = str(request.form['date'])
    #datetime.strptime(request.form['date'], '%Y-%m-%d')

    # Model columns:
    # ['SCHEDULED_DEPARTURE','DATE','SCHEDULED_ARRIVAL','AIRLINE_NAME_Southwest Airlines Co.',
    # 'AIRLINE_NAME_Delta Air Lines Inc.','AIRLINE_NAME_Spirit Air Lines','MONTH_6','AIRLINE_NAME_Alaska Airlines Inc.',
    # 'MONTH_2','AIRLINE_NAME_JetBlue Airways','ORIGIN_AC_ORD','DEST_STATE_NY','DESTINATION_AC_LGA',
    # 'ORIGIN_STATE_IL','ORIGIN_AC_DFW','ORIGIN_AC_SEA']

    # prepare the data
    ## 1. Load datasets
    df_airports = pd.read_csv('data/airports.csv')

    ## 2. Extract & define features
    month = datetime.strptime(date, '%Y-%m-%d').month
    new_sched_arr = time_to_num(sched_arr)
    new_sched_dep = time_to_num(sched_dep)
    new_date = date_to_int(date)
    dest_state = df_airports[df_airports['IATA_CODE'] == dest_airport]['STATE'].values[0]
    origin_state = df_airports[df_airports['IATA_CODE'] == origin_airport]['STATE'].values[0]

    ## 3. Identify the values of dummy columns
    Southwest_Airlines_Co, Delta_Air_Lines_Inc, Spirit_Air_Lines, Alaska_Airlines_Inc, JetBlue_Airways = get_airline_columns(airline)
    MONTH_6, MONTH_2 = get_month_columns(month)
    ORIGIN_AC_ORD, ORIGIN_AC_DFW, ORIGIN_AC_SEA = get_origin_AC_columns(origin_airport)

    if (dest_airport == 'LGA'):
        DESTINATION_AC_LGA = 1
    else:
        DESTINATION_AC_LGA = 0

    if (origin_state == 'IL'):
        ORIGIN_STATE_IL = 1
    else:
        ORIGIN_STATE_IL = 0

    if (dest_state == 'NY'):
        DEST_STATE_NY = 1
    else:
        DEST_STATE_NY = 0

    model_values = [new_sched_arr, new_date, new_sched_dep,Southwest_Airlines_Co, Delta_Air_Lines_Inc, Spirit_Air_Lines,
                    MONTH_6, Alaska_Airlines_Inc, MONTH_2, JetBlue_Airways, ORIGIN_AC_ORD,DEST_STATE_NY,DESTINATION_AC_LGA,
                    ORIGIN_STATE_IL,ORIGIN_AC_DFW, ORIGIN_AC_SEA]

    # load the model and predict
    with open('model/gbm.pkl', 'rb') as fid:
        gb_model = cPickle.load(fid)
        
    prediction = gb_model.predict([model_values])
    predicted_delay = prediction.round(2)[0]

    return render_template('results.html',
                           origin_airport= (origin_airport),
                           dest_airport= (dest_airport),
                           airline = (airline),
                           distance= (distance),
                           sched_arr= (sched_arr),
                           sched_dep= (sched_dep),
                           date = (date),
                           predicted_delay="{:,}".format(predicted_delay)
                           )


# convert date to timestamp 
def date_to_int(time_str):
    if type(time_str) is float:
        return time_str
    
    result = datetime.fromisoformat(time_str).timestamp()
    return result

# Convert the time to seconds
def time_to_num (time_str):
    if type(time_str) is int:
        return time_str
    
    h,m = time_str.split(':')
    result = int(h) * 3600 + int(m) * 60
    return result

def get_airline_columns(airline):

    if (airline == 'Southwest Airlines Co.'):
        AIRLINE_NAME_Southwest_Airlines_Co = 1
        AIRLINE_NAME_Delta_Air_Lines_Inc = 0
        AIRLINE_NAME_Spirit_Air_Lines = 0
        AIRLINE_NAME_Alaska_Airlines_Inc = 0
        AIRLINE_NAME_JetBlue_Airways = 0

    elif (airline == 'Delta Air Lines Inc.'):
        AIRLINE_NAME_Southwest_Airlines_Co = 0
        AIRLINE_NAME_Delta_Air_Lines_Inc = 1
        AIRLINE_NAME_Spirit_Air_Lines = 0
        AIRLINE_NAME_Alaska_Airlines_Inc = 0
        AIRLINE_NAME_JetBlue_Airways = 0

    elif (airline == 'Spirit Air Lines'):
        AIRLINE_NAME_Southwest_Airlines_Co = 0
        AIRLINE_NAME_Delta_Air_Lines_Inc = 0
        AIRLINE_NAME_Spirit_Air_Lines = 1
        AIRLINE_NAME_Alaska_Airlines_Inc = 0
        AIRLINE_NAME_JetBlue_Airways = 0

    elif (airline == 'Alaska Airlines Inc.'):
        AIRLINE_NAME_Southwest_Airlines_Co = 0
        AIRLINE_NAME_Delta_Air_Lines_Inc = 0
        AIRLINE_NAME_Spirit_Air_Lines = 0
        AIRLINE_NAME_Alaska_Airlines_Inc = 1
        AIRLINE_NAME_JetBlue_Airways = 0
    
    elif (airline == 'JetBlue Airways'):
        AIRLINE_NAME_Southwest_Airlines_Co = 0
        AIRLINE_NAME_Delta_Air_Lines_Inc = 0
        AIRLINE_NAME_Spirit_Air_Lines = 0
        AIRLINE_NAME_Alaska_Airlines_Inc = 0
        AIRLINE_NAME_JetBlue_Airways = 1


    else:
        AIRLINE_NAME_Southwest_Airlines_Co = 0
        AIRLINE_NAME_Delta_Air_Lines_Inc = 0
        AIRLINE_NAME_Spirit_Air_Lines = 0
        AIRLINE_NAME_Alaska_Airlines_Inc = 0
        AIRLINE_NAME_JetBlue_Airways = 0

    return AIRLINE_NAME_Southwest_Airlines_Co, AIRLINE_NAME_Delta_Air_Lines_Inc, AIRLINE_NAME_Spirit_Air_Lines, AIRLINE_NAME_Alaska_Airlines_Inc, AIRLINE_NAME_JetBlue_Airways

def get_month_columns(month):

    if (month == 6):
        MONTH_6 = 1
        MONTH_2 = 0

    elif (month == 2):
        MONTH_6 = 0
        MONTH_2 = 1
    
    else:
        MONTH_6 = 0
        MONTH_2 = 0

    return MONTH_6, MONTH_2

def get_origin_AC_columns(airport_code):
    if (airport_code == 'ORD'):
        ORD = 1
        DFW = 0
        SEA = 0        

    elif (airport_code == 'DFW'):
        ORD = 0
        DFW = 1
        SEA = 0

    elif (airport_code == 'SEA'):
        ORD = 0
        DFW = 0
        SEA = 1

    return ORD, DFW, SEA

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
