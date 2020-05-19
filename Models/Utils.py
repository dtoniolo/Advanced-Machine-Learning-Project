import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_history(history):
  epochs = history.epoch
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
    fig.add_trace(go.Scatter(x=epochs, y=history['val_' + key], mode = 'lines',
                             name='Validation', legendgroup='Validation',
                              marker_color='#FC766A', showlegend=showlegend), row=i , col=1)
    fig.update_yaxes(title=key.capitalize(), row=i, col=1)
  fig.update_layout(template='plotly_white', height = 300*n_plots)
  fig.update_xaxes(title='Epochs')
  return fig
