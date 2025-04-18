import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash
from sklearn.linear_model import LinearRegression
import joblib
import os

class EDA:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.result_df = None

    def summary_statistics(self):
        summaries = {}
        for sector, df in self.data_dict.items():
            summaries[sector] = df.drop(columns=['Year'], errors='ignore').describe()
        return summaries

    def missing_values(self):
        missing_data = {}
        for sector, df in self.data_dict.items():
            missing_data[sector] = df.drop(columns=['Year'], errors='ignore').isnull().sum()
        return missing_data

    def correlation_matrix(self):
        macro_df = self.data_dict.get('Macroeconomic Indicators')
        combined_df = macro_df.drop(columns=['Year'], errors='ignore') if macro_df is not None else pd.DataFrame()

        for sector, df in self.data_dict.items():
            if sector == 'Macroeconomic Indicators':
                continue
            sector_df = df.drop(columns=['Year'], errors='ignore')
            sector_df.columns = [f"{sector}_{col}" for col in sector_df.columns]
            combined_df = pd.concat([combined_df, sector_df], axis=1)

        return combined_df.corr()

    def run_pipeline(self):
        self.result_df = {
            "summary_statistics": self.summary_statistics(),
            "missing_values": self.missing_values(),
            "correlation_matrix": self.correlation_matrix()
        }
        return self.result_df