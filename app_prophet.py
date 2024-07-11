from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the trained Prophet model
model = joblib.load('prophet_model.pkl')

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    timestamp = request.form['timestamp']
    days = int(request.form['days'])
    
    # Convert input timestamp to datetime
    start_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    
    # Create future dataframe from the given timestamp
    end_time = start_time + timedelta(days=days)
    future = pd.date_range(start=start_time, end=end_time, freq='3min')
    future_df = pd.DataFrame({'ds': future})

    # Make predictions
    forecast = model.predict(future_df)
    
    # Sum the predictions over the period
    total_prediction = forecast['yhat'].sum()
    
    result = {'total_prediction': total_prediction}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
