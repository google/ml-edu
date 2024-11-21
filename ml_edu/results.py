# Copyright 2024 The ml_edu Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plotting utitlies to visualize experiment results and compare them."""

import matplotlib.lines
import matplotlib.pyplot as plt
import ml_edu.experiment as ml_experiment
import numpy as np
import pandas as pd


def plot_experiment_metrics(
    experiment: ml_experiment.Experiment, metrics: list[str]
):
  """Plot a curve of one or more metrics for different epochs."""
  plt.figure()

  for metric in metrics:
    plt.plot(
        experiment.epochs, experiment.metrics_history[metric], label=metric
    )

  plt.xlabel('Epoch')
  plt.ylabel('Metric value')
  plt.grid()
  plt.legend()


def compare_experiment(
    experiments: list[ml_experiment.Experiment],
    metrics_of_interest: list[str],
    test_dataset: pd.DataFrame,
    test_labels: np.ndarray,
):
  """Compare the metrics of multiple experiments by plotting them together."""
  # Make sure that we have all the data we need.
  for metric in metrics_of_interest:
    for experiment in experiments:
      if metric not in experiment.metrics_history:
        raise ValueError(
            f'Metric {metric} not available for experiment {experiment.name}'
        )

  fig = plt.figure(figsize=(12, 12))
  ax = fig.add_subplot(2, 1, 1)

  colors = [f'C{i}' for i in range(len(experiments))]
  markers = ['.', '*', 'd', 's', 'p', 'x']
  marker_size = 10

  ax.set_title('Train metrics')
  for i, metric in enumerate(metrics_of_interest):
    for j, experiment in enumerate(experiments):
      ax.plot(
          experiment.epochs,
          experiment.metrics_history[metric],
          markevery=4,
          marker=markers[i],
          markersize=marker_size,
          color=colors[j],
      )

  # Add custom legend to show what the colors and markers mean
  legend_handles = []
  for i, metric in enumerate(metrics_of_interest):
    legend_handles.append(
        matplotlib.lines.Line2D(
            [0],
            [0],
            label=metric,
            marker=markers[i],
            markersize=marker_size,
            c='k',
        )
    )
  for i, experiment in enumerate(experiments):
    legend_handles.append(
        matplotlib.lines.Line2D(
            [0], [0], label=experiment.name, color=colors[i]
        )
    )

  ax.set_xlabel('Epoch')
  ax.set_ylabel('Metric value')
  ax.grid()
  ax.legend(handles=legend_handles)

  ax = fig.add_subplot(2, 1, 2)
  spacing = 0.3
  n_bars = len(experiments)
  bar_width = (1 - spacing) / n_bars
  for i, experiment in enumerate(experiments):
    test_metrics = experiment.evaluate(test_dataset, test_labels)
    x = np.arange(len(metrics_of_interest)) + bar_width * (
        i + 1 / 2 - n_bars / 2
    )
    ax.bar(
        x,
        [test_metrics[metric] for metric in metrics_of_interest],
        width=bar_width,
        label=experiment.name,
    )
  ax.set_xticks(np.arange(len(metrics_of_interest)), metrics_of_interest)

  ax.set_title('Test metrics')
  ax.set_ylabel('Metric value')
  ax.set_axisbelow(True)  # Put the grid behind the bars
  ax.grid()
  ax.legend()
