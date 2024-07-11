from flask import Flask, request, render_template
import joblib
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from multiprocessing import Pool, cpu_count

app = Flask(__name__)

# Global variables for model and scalers
global lstm_model, scaler_features, scaler_target

# Load the model and scalers
lstm_model = joblib.load('lstm_model1.pkl')
scaler_features = joblib.load('scaler_features.pkl')
scaler_target = joblib.load('scaler_target.pkl')

def predict_kwh(features_scaled, lstm_model, scaler_target):
    features_array = np.expand_dims(features_scaled, axis=0)
    features_array = np.repeat(features_array, 10, axis=1)
    
    kwh_prediction_scaled = lstm_model.predict(features_array)
    kwh_prediction = scaler_target.inverse_transform(kwh_prediction_scaled)
    
    return kwh_prediction[0][0]

@app.route('/')
def index():
    return render_template('index.html', previous_inputs=None, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        voltage = request.form['voltage']
        if not voltage:
            voltage = float(251)
        else:
            voltage = float(voltage)
        
        start_timestamp = request.form['start_timestamp']
        stop_timestamp = request.form['stop_timestamp']
        start_datetime = datetime.strptime(start_timestamp, '%Y-%m-%dT%H:%M')
        stop_datetime = datetime.strptime(stop_timestamp, '%Y-%m-%dT%H:%M')
        
        if stop_datetime <= start_datetime:
            return render_template('index.html', prediction="Stop timestamp must be after start timestamp.")
        
        devices = request.form.getlist('device[]')
        wattage_defaults = request.form.getlist('wattage_default[]')
        wattages = [float(w) if w != 'custom' else float(c) for w, c in zip(wattage_defaults, request.form.getlist('wattage[]'))]
        num_devices = list(map(int, request.form.getlist('num_devices[]')))
    
        month = start_datetime.month
        if 'AC' in devices and (month >= 4 and month <= 7):  # Summer impact for AC
            impact_factor = 1.5  
            ac_index = devices.index('AC')
            ac_wattage = wattages[ac_index] * impact_factor
            wattages[ac_index] = ac_wattage
        elif 'HEATER' in devices and (month <= 2 or month == 12 or month == 11):  # Winter impact for Heater
            impact_factor = 1.5  
            heater_index = devices.index('HEATER')
            heater_wattage = wattages[heater_index] * impact_factor
            wattages[heater_index] = heater_wattage
        else:
            impact_factor = 1.0  # Default impact
        
        net_current_value = sum((wattage / voltage) * num for wattage, num in zip(wattages, num_devices))

        date_range = pd.date_range(start=start_datetime, end=stop_datetime, freq='min')
        total_kwh = parallel_predict(date_range, voltage, net_current_value)
        
        previous_inputs = {
            'voltage': voltage,
            'start_timestamp': start_timestamp,
            'stop_timestamp': stop_timestamp,
            'devices': devices,
            'wattage_defaults': wattage_defaults,
            'wattages': wattages,
            'num_devices': num_devices
        }

        return render_template('index.html', prediction=(total_kwh/10), previous_inputs=previous_inputs)
    
    except Exception as e:
        return str(e)

def get_features(voltage, net_current_value, current_datetime):
    minute = current_datetime.minute
    week = current_datetime.isocalendar()[1]
    month = current_datetime.month
    day = current_datetime.day
    return [voltage, net_current_value, minute, week, month, day]

def single_prediction(args):
    voltage, net_current_value, current_datetime = args
    features = get_features(voltage, net_current_value, current_datetime)
    features_scaled = scaler_features.transform([features])
    prediction = predict_kwh(features_scaled, lstm_model, scaler_target)
    return prediction

def parallel_predict(date_range, voltage, net_current_value):
    args_list = [(voltage, net_current_value, dt) for dt in date_range]
    with Pool(cpu_count(), initializer=init_worker) as pool:
        predictions = pool.map(single_prediction, args_list)
    return sum(predictions)

def init_worker():
    global lstm_model, scaler_features, scaler_target
    lstm_model = joblib.load('lstm_model1.pkl')
    scaler_features = joblib.load('scaler_features.pkl')
    scaler_target = joblib.load('scaler_target.pkl')

if __name__ == '__main__':
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    app.run(debug=True)
