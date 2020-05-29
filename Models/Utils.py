import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_history(history):
    epochs = history.epoch
    epochs = np.array(epochs) +1
    history = history.history
    n_plots = int(len(history)/2)
    plots = list()
    for key in history.keys():
        if 'val_' not in key and key not in plots:
            plots.append(key)
    fig = make_subplots(rows = n_plots, cols = 1, shared_xaxes=True,
                        subplot_titles=[name.capitalize() for name in plots])
    for i, key in enumerate(plots):
        if i == 1:
            showlegend=True
        else:
            showlegend=False
        i += 1
        fig.add_trace(go.Scatter(x=epochs, y=history[key], mode = 'lines',
                              name='Training', legendgroup='Training',
                              marker_color='#5B84B1', showlegend=showlegend), row=i , col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history['val_'+key], mode ='lines',
                              name='Validation', legendgroup='Validation',
                              marker_color='#FC766A', showlegend=showlegend), row=i , col=1)
        fig.update_yaxes(title=key.capitalize(), row=i, col=1)
    fig.update_layout(template='plotly_white', height = 300*n_plots)
    fig.update_xaxes(title='Epochs', range=[0, epochs[-1]+1])
    for i in range(1, n_plots+1):
        fig.update_xaxes(showticklabels=True, row=i, col=1)
    return fig

def plot_history_hpo(results):
    quantities = results.values[0].history.keys()
    plots = list()
    for key in quantities:
        if 'val_' not in key:
            plots.append(key)
    n_plots = int(len(quantities)/2)
    fig = make_subplots(rows = n_plots, cols = 1, shared_xaxes=True,
                        subplot_titles=[name.capitalize() for name in plots])
    for i, key in enumerate(plots):
        if i == 1:
          showlegend=True
        else:
          showlegend=False
        i += 1
        for name in results:
            history = results[name].history
            epochs = history.epochs
            fig.add_trace(go.Scatter(x=epochs, y=history[key], mode = 'lines',
                                     name='Training'+name, legendgroup='Training', showlegend=showlegend), row=i , col=1)
            fig.add_trace(go.Scatter(x=epochs, y=history['val_' + key], mode = 'lines',
                                     name='Validation'+name, legendgroup='Validation', showlegend=showlegend), row=i , col=1)
        fig.update_yaxes(title=key.capitalize(), row=i, col=1)
    fig.update_layout(template='plotly_white', height = 300*n_plots)
    fig.update_xaxes(title='Epochs')
    return fig
