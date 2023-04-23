from flask import Flask, render_template, request
import pickle
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

# Load the trained model
with open('Air_Pollution_web_on_pickle.pkl', 'rb') as file:
    model = pickle.load(file)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Predict function
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input value from user
    forecast_period = int(request.form['forecast_period'])

    # Load the dataset
    df = pd.read_csv('city_day.csv')

    # Preprocess the data
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.dropna(subset=['AQI'])
    df = df.resample('D').mean()

    # Split the data into train and test sets
    train = df.iloc[:-365]
    test = df.iloc[-365:]

    # Train the SARIMAX model
    model = SARIMAX(train['AQI'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_invertibility=False)
    model_fit = model.fit()

    # Generate the predicted AQI values
    if forecast_period < 24:
        forecasts = model_fit.forecast(steps=forecast_period)
        date_range = pd.date_range(start=train.index[-1], periods=forecast_period, freq='H')
    else:
        forecasts = model_fit.forecast(steps=int(forecast_period / 24))
        date_range = pd.date_range(start=train.index[-1], periods=int(forecast_period / 24), freq='D')
        forecasts = forecasts.resample('H').ffill()

    # Format the output data into a DataFrame
    output_data = pd.DataFrame({'Date': date_range[1], 'Predicted AQI': forecasts})
    output_data = output_data[['Date', 'Predicted AQI']]
    # Render the output data to HTML
    return render_template('index.html', tables=[output_data.to_html(classes='data')], titles=output_data.columns.values)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
