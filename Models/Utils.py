import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import tensorflow as tf


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
    fig.update_layout(template='plotly_white', height = 400*n_plots)
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


def get_confusion_matrix(model, data_generator):
    y_hat = model.predict(data_generator)
    y_hat = np.argmax(y_hat, axis=1)
    return confusion_matrix(data_generator.classes, y_hat)

from google.colab import files
import itertools
import matplotlib.pyplot as plt
def plot_cm(conf_mtx, labels, normalize=False, cmap=plt.cm.Reds):
  if normalize:
    conf_mtx = conf_mtx.astype('float') / conf_mtx.sum(axis=1)[:, np.newaxis]
    print("Normalized Confusion Matrix")
  else:
    print("Confusion Matrix")
  leftmargin = 0.5 # inches
  rightmargin = 0.5 # inches
  categorysize = 0.5 # inches
  figwidth = leftmargin + rightmargin + (len(labels) * categorysize)

  f = plt.figure(figsize=(figwidth,figwidth))
  ax = f.add_subplot(111)
  ax.set_aspect(1)
  f.subplots_adjust(left=leftmargin/figwidth, right=1-rightmargin/figwidth, top=0.94, bottom=0.1)

  res = ax.imshow(conf_mtx, interpolation='nearest', cmap=cmap)
  plt.title("Confusion Matrix")
  plt.colorbar(res)
  ax.set_xticks(range(len(labels)))
  ax.set_yticks(range(len(labels)))
  ax.set_xticklabels(labels, rotation=45, ha='right')
  ax.set_yticklabels(labels)

  fmt = '.2f' if normalize else 'd'
  thresh = conf_mtx.max() / 2.
  for i, j in itertools.product(range(conf_mtx.shape[0]), range(conf_mtx.shape[1])):
    plt.text(j, i, format(conf_mtx[i, j], fmt),
    horizontalalignment="center",
    color="white" if conf_mtx[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig("confusion_matrixVGG16.png")
  return f
