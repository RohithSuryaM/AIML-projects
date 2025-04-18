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

class Dashboard:
    def __init__(self, data_dict, eda_results, forecast_results, regression_results):
        self.data_dict = data_dict
        self.eda_results = eda_results
        self.forecast_results = forecast_results
        self.regression_results = regression_results
        self.app = Dash(__name__)
        self.layout()

    def layout(self):
        self.app.layout = html.Div([
            html.H1("Employment & Economic Dashboard"),
            dcc.Tabs([
                dcc.Tab(label='EDA Summary', children=[
                    dcc.Graph(figure=px.imshow(
                        self.eda_results['correlation_matrix'],
                        title="Correlation Matrix: Employment vs Macroeconomic Indicators",
                        labels=dict(color="Correlation"),
                        color_continuous_scale="RdBu_r"
                    ))
                ]),
                dcc.Tab(label='Forecasting', children=[
                    dcc.Graph(id='forecast_graph'),
                    dcc.Dropdown(id='sector_dropdown',
                                 options=[{'label': k, 'value': k} for k in self.forecast_results.keys()],
                                 value=list(self.forecast_results.keys())[0])
                ]),
                dcc.Tab(label='Regression Analysis', children=[
                    dcc.Graph(id='regression_graph'),
                    dcc.Dropdown(id='regression_sector_dropdown',
                                 options=[{'label': k, 'value': k} for k in self.regression_results['normal'].keys()],
                                 value=list(self.regression_results['normal'].keys())[0])
                ]),
                dcc.Tab(label='Reverse Regression', children=[
                    dcc.Graph(id='reverse_regression_graph'),
                    dcc.Dropdown(id='reverse_regression_sector_dropdown',
                                 options=[{'label': k, 'value': k} for k in self.regression_results['reverse'].keys()],
                                 value=list(self.regression_results['reverse'].keys())[0])
                ])
            ])
        ])

        @self.app.callback(
            Output('forecast_graph', 'figure'),
            [Input('sector_dropdown', 'value')]
        )
        def update_forecast_graph(sector):
            df = self.forecast_results.get(sector)
            if df is None:
                return go.Figure().update_layout(title="No forecast available.")

            fig = go.Figure()
            for col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
            fig.update_layout(
                title=f"Forecast for {sector}",
                xaxis_title="Index",
                yaxis_title="Forecasted Employment Values"
            )
            return fig

        @self.app.callback(
            Output('regression_graph', 'figure'),
            [Input('regression_sector_dropdown', 'value')]
        )
        def update_regression_graph(sector):
            macro_df = self.data_dict['Macroeconomic Indicators'].drop(columns=['Year'], errors='ignore')
            y = self.data_dict[sector].drop(columns=['Year'], errors='ignore')

            model_path = os.path.join('models', f"regression_{sector}.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
            else:
                model = LinearRegression()
                model.fit(macro_df, y)

            predictions = model.predict(macro_df)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=macro_df.index, y=y.mean(axis=1), mode='markers', name='Actual Employment'))
            fig.add_trace(go.Scatter(x=macro_df.index, y=predictions.mean(axis=1), mode='lines', name='Predicted Employment'))
            fig.update_layout(
                title=f"Linear Regression Fit for {sector}",
                xaxis_title="Index",
                yaxis_title="Employment Values"
            )
            return fig

        @self.app.callback(
            Output('reverse_regression_graph', 'figure'),
            [Input('reverse_regression_sector_dropdown', 'value')]
        )
        def update_reverse_regression_graph(sector):
            employment_df = self.data_dict[sector].drop(columns=['Year'], errors='ignore')
            y = self.data_dict['Macroeconomic Indicators'].drop(columns=['Year'], errors='ignore')

            model = LinearRegression()
            model.fit(employment_df, y)

            predictions = model.predict(employment_df)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=employment_df.index, y=y.mean(axis=1), mode='markers', name='Actual Macro'))
            fig.add_trace(go.Scatter(x=employment_df.index, y=predictions.mean(axis=1), mode='lines', name='Predicted Macro'))
            fig.update_layout(
                title=f"Reverse Regression Fit: {sector} predicting Macro Indicators",
                xaxis_title="Index",
                yaxis_title="Macroeconomic Values"
            )
            return fig
    
    def run(self):
        self.app.run(mode='inline', port=8083)