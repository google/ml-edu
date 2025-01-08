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

import keras
from ml_edu import experiment
import numpy as np
import pandas as pd


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
          classification_threshold=0,
          input_features=[],
      ),
      model=keras.Model(),
      epochs=np.array([]),
      metrics_history=history,
  )
  assert exp.get_final_metric_value("loss") == 0.3
  assert exp.get_final_metric_value("accuracy") == 0.6
