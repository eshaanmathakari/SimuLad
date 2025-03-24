# simulation.py
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

def convert_temperature_to_fahrenheit(data):
    """
    Converts temperature columns (containing "Temp" and "degC") from Celsius to Fahrenheit.
    F = C * 1.8 + 32.
    Renames the column to indicate degF.
    """
    data = data.copy()
    for col in data.columns:
        if "temp" in col.lower() and "degc" in col.lower():
            data[col] = data[col] * 1.8 + 32
            new_col = col.replace("degC", "degF").replace("DEGC", "degF")
            data.rename(columns={col: new_col}, inplace=True)
    return data

def train_var_model(data, maxlags=3):
    """
    Trains a VAR model on the given DataFrame.
    Converts temperature to Fahrenheit.
    If the number of observations is too small, automatically reduces maxlags.
    If a LinAlgError occurs (due to near-singular covariance), differences the data.
    
    Returns:
      - results: the fitted VAR model,
      - last_level: None if the level model was used, or the last level observation if differencing was used.
    """
    # Convert temperature columns to Fahrenheit.
    data_converted = convert_temperature_to_fahrenheit(data)
    data_indexed = data_converted.set_index("DateTime")
    n_obs = len(data_indexed)
    # Adjust maxlags if necessary.
    if n_obs <= maxlags:
        maxlags = max(1, n_obs - 1)
    
    try:
        model = VAR(data_indexed)
        lag_order_results = model.select_order(maxlags)
        lag = lag_order_results.aic if lag_order_results.aic is not None else 1
        results = model.fit(lag)
        return results, None  # Level model used.
    except np.linalg.LinAlgError as err:
        # Try differencing the data.
        data_diff = data_indexed.diff().dropna()
        if len(data_diff) <= maxlags:
            maxlags = max(1, len(data_diff) - 1)
        model = VAR(data_diff)
        lag_order_results = model.select_order(maxlags)
        lag = lag_order_results.aic if lag_order_results.aic is not None else 1
        results = model.fit(lag)
        last_level = data_indexed.iloc[-1]
        return results, last_level

def simulate_scenario(data, var_results, adjustments, steps=24, last_level=None):
    """
    Simulates a scenario given:
      - data: original DataFrame (with DateTime column).
      - var_results: fitted VAR model.
      - adjustments: dict of variable adjustments (e.g., {"RainForest_MountainTower_Temp": 0.05}).
      - steps: forecast horizon (default 24 hours).
      - last_level: if not None, indicates that differenced data were used.
      
    Returns a forecast DataFrame.
    """
    data_converted = convert_temperature_to_fahrenheit(data)
    data_indexed = data_converted.set_index("DateTime")
    
    if last_level is None:
        # Level model: use last k_ar observations.
        last_obs = data_indexed.iloc[-var_results.k_ar:]
        modified_obs = last_obs.copy()
        for var, change in adjustments.items():
            if var in modified_obs.columns:
                modified_obs.iloc[-1, modified_obs.columns.get_loc(var)] += change
        forecast = var_results.forecast(modified_obs.values, steps=steps)
        forecast_index = pd.date_range(start=data["DateTime"].iloc[-1], periods=steps+1, freq='H')[1:]
        forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=last_obs.columns)
        return forecast_df
    else:
        # Differenced model: forecast differences and reconstruct levels.
        data_diff = data_indexed.diff().dropna()
        last_obs_diff = data_diff.iloc[-var_results.k_ar:]
        modified_obs_diff = last_obs_diff.copy()
        # Compute adjustment in levels.
        new_last = last_level.copy()
        for var, change in adjustments.items():
            if var in new_last.index:
                new_last[var] += change
        original_last = data_indexed.iloc[-1]
        diff_adjustment = new_last - original_last
        # Apply adjustment to the last row of differenced data.
        modified_obs_diff.iloc[-1] = modified_obs_diff.iloc[-1] + diff_adjustment
        forecast_diff = var_results.forecast(modified_obs_diff.values, steps=steps)
        forecast_diff_df = pd.DataFrame(forecast_diff, columns=data_indexed.columns)
        # Reconstruct level forecasts by cumulative sum.
        forecast_level = forecast_diff_df.cumsum() + last_level.values
        forecast_index = pd.date_range(start=data["DateTime"].iloc[-1], periods=steps+1, freq='H')[1:]
        forecast_df = pd.DataFrame(forecast_level, index=forecast_index, columns=data_indexed.columns)
        return forecast_df
