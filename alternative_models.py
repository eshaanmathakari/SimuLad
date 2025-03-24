# alternative_models.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

def forecast_arima(data, order=(1,1,1), steps=24):
    """
    Trains an ARIMA model on a univariate time series (assumes the second column is the measurement)
    and forecasts the next 'steps' hours.
    """
    ts = data.set_index("DateTime").iloc[:, 1]
    if len(ts) < 2:
        raise ValueError("Not enough data for ARIMA forecasting")
    model = ARIMA(ts, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    forecast_df = forecast.to_frame(name=ts.name)
    forecast_df.index = pd.date_range(start=data["DateTime"].iloc[-1], periods=steps+1, freq='H')[1:]
    return forecast_df

def forecast_prophet(data, steps=24):
    """
    Trains a Prophet model on a univariate time series and forecasts the next 'steps' hours.
    Prophet expects a DataFrame with columns 'ds' (datetime) and 'y' (measurement).
    """
    ts = data.set_index("DateTime").iloc[:, 1].reset_index()
    ts.rename(columns={"DateTime": "ds", ts.columns[1]: "y"}, inplace=True)
    if ts["y"].dropna().shape[0] < 2:
        raise ValueError("Not enough non-NaN data for Prophet forecasting")
    m = Prophet()
    m.fit(ts)
    future = m.make_future_dataframe(periods=steps, freq='H')
    forecast = m.predict(future)
    forecast_df = forecast[["ds", "yhat"]].tail(steps)
    forecast_df.set_index("ds", inplace=True)
    return forecast_df
