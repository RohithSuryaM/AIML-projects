import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash
import io
import base64

class TimeSeriesForecasting:
    def __init__(self, data_dict, forecast_years=10):
        self.data_dict = data_dict
        self.forecast_years = forecast_years
        self.forecast_results = {}
    
    def apply_forecasting(self, df):
        df = df.dropna()
        if len(df) < 2:
            return None  # Not enough data
        df = df.set_index(df.columns[0])
        forecast_df = pd.DataFrame()
        
        for col in df.columns:
            model = ExponentialSmoothing(df[col], trend='add', seasonal='add', seasonal_periods=5).fit()
            pred = model.forecast(self.forecast_years)
            forecast_df[col] = pred
        return forecast_df
    
    def forecast_all_sectors(self):
        for sector, df in self.data_dict.items():
            self.forecast_results[sector] = self.apply_forecasting(df)
    
    def run_pipeline(self):
        self.forecast_all_sectors()
        return self.forecast_results