# Copyright 2025 The ml_edu Authors.
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

import keras
from ml_edu import experiment
import numpy as np
import pandas as pd


def test_evaluate():
  settings = experiment.ExperimentSettings(
      learning_rate=0.01,
      number_epochs=1,
      batch_size=1,
      classification_threshold=0,
      input_features=["input_feature"],
  )
  model_input = keras.Input(name="input_feature", shape=(1,))
  model_output = keras.layers.Dense(
      units=1,
      name="dense_layer",
      activation=keras.activations.sigmoid,
  )(model_input)
  model = keras.Model(inputs=model_input, outputs=model_output)
  model.compile(
      optimizer=keras.optimizers.RMSprop(settings.learning_rate),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=["precision", "recall", "accuracy"],
  )
  history = model.fit(
      x=pd.DataFrame({"input_feature": [0, 1, 2, 3]}),
      y=np.array([0, 1, 0, 1]),
      batch_size=settings.batch_size,
      epochs=settings.number_epochs,
  )
  exp = experiment.Experiment(
      name="test_experiment",
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )
  evaluation_results = exp.evaluate(
      pd.DataFrame({"input_feature": [1.5, 2.5]}), np.array([0, 1])
  )
  # Actual results are not deterministic, so we only check that the expected
  # keys are present.
  assert set(evaluation_results.keys()) == {
      "loss",
      "precision",
      "recall",
      "accuracy",
  }


def test_get_final_metric_value():
  history = pd.DataFrame({
      "loss": [0.1, 0.2, 0.3],
      "accuracy": [0.4, 0.5, 0.6],
  })
  exp = experiment.Experiment(
      name="test_experiment",
      settings=experiment.ExperimentSettings(
          learning_rate=0,
          number_epochs=0,
          batch_size=0,
          input_features=[],
      ),
      model=keras.Model(),
      epochs=np.array([]),
      metrics_history=history,
  )
  assert exp.get_final_metric_value("loss") == 0.3
  assert exp.get_final_metric_value("accuracy") == 0.6


def test_regression_settings():
  settings = experiment.ExperimentSettings(
      learning_rate=0.01,
      number_epochs=1,
      batch_size=1,
      input_features=["input_feature"],
  )
  assert settings.classification_threshold is None
