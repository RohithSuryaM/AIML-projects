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

class MultipleLinearRegression:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.models = {}
        self.reverse_models = {}  # For employment as independent variables
        self.results = {}
        self.reverse_results = {}

    def train_models(self):
        macro_df = self.data_dict.get('Macroeconomic Indicators')
        if macro_df is None:
            return None
        X = macro_df.drop(columns=['Year'], errors='ignore')
        for sector, df in self.data_dict.items():
            if sector == 'Macroeconomic Indicators':
                continue
            y = df.drop(columns=['Year'], errors='ignore')
            model = LinearRegression()
            model.fit(y, X)
            self.models[sector] = model
            self.results[sector] = model.coef_

    def train_reverse_models(self):
        macro_df = self.data_dict.get('Macroeconomic Indicators')
        if macro_df is None:
            return None
        y = macro_df.drop(columns=['Year'], errors='ignore')
        for sector, df in self.data_dict.items():
            if sector == 'Macroeconomic Indicators':
                continue
            X = df.drop(columns=['Year'], errors='ignore')
            model = LinearRegression()
            model.fit(X, y)
            self.reverse_models[sector] = model
            self.reverse_results[sector] = model.coef_

    def run_pipeline(self):
        self.train_models()
        self.train_reverse_models()
        return {
            "normal": self.results,
            "reverse": self.reverse_results
        }