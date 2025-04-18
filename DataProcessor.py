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

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.processed_data = {}
    
    def run_pipeline(self):
        self.clean_and_transform_data()
        return self.processed_data

    def clean_and_transform_data(self):
        xls = pd.ExcelFile(self.file_path)
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            df = df.dropna(how='all')  # Remove empty rows
            df = df.dropna(axis=1, how='all')  # Remove empty columns
            df = df.ffill().bfill()  # Fill missing values
            df.columns = df.columns.str.strip()  # Remove column name whitespace
            df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
            df = self.convert_financial_year(df)
            df = self.convert_datetime(df)
            df = self.encode_categorical(df)
            self.processed_data[sheet] = df

    def convert_financial_year(self, df):
        if 'Year' in df.columns:
            df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')[0].astype(float).astype('Int64')
        return df

    def convert_datetime(self, df):
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    continue
        return df

    def encode_categorical(self, df):
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() < 10:
                df = pd.get_dummies(df, columns=[col])
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        return df