import plotly.graph_objs as go
import plotly.express as px
import numpy as np


def plot_results(Y_test, Y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(Y_test))), y=Y_test, mode='lines+markers', name='Actual', line=dict(color='royalblue', width=2), marker=dict(color='royalblue', size=5)))
    fig.add_trace(go.Scatter(x=list(range(len(Y_test))), y=Y_pred, mode='lines+markers', name='Predicted', line=dict(color='firebrick', width=2), marker=dict(color='firebrick', size=5)))
    fig.update_layout(title='Actual vs Predicted Housing Prices', xaxis_title='Index', yaxis_title='Price', template='plotly_white')
    fig.show()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    fig = go.Figure(go.Bar(x=[feature_names[i] for i in indices], y=[importances[i] for i in indices], marker=dict(color='rgba(50, 171, 96, 0.6)', line=dict(color='rgba(50, 171, 96, 1.0)', width=1))))
    fig.update_layout(title='Feature Importance', xaxis_title='Features', yaxis_title='Importance', template='plotly_white')
    fig.show()

def plot_residuals(Y_test, Y_pred):
    residuals = Y_test - Y_pred
    fig = px.scatter(x=Y_test, y=residuals, title="Residuals Plot", labels={'x': 'Actual Prices', 'y': 'Residuals'})
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.show()