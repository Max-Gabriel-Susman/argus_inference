# Copyright 2026 Max Gabriel Susman
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

"""ARGUS_MODEL_PATH: saved-model loading and the [counts, power] vector."""

import pickle

from argus_core.msg import NeuralFrame
from argus_inference.inference_node import _check_model_bundle, _frame_features
import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAYOUT = {'features': ['counts', 'power'], 'channels': 96, 'bin_len': 1500, 'mult': 3.5}


def _frame(channel_count=96):
    msg = NeuralFrame()
    msg.channel_count = channel_count
    msg.channels = [i % 5 for i in range(96)]
    msg.power = [1000 + 10 * i for i in range(96)]
    return msg


def _fitted(n_features):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, n_features))
    y = np.arange(80) % 4
    X[np.arange(80), y] += 5.0
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()).fit(X, y)


def test_vector_is_counts_then_power():
    x = _frame_features(_frame(), ['counts', 'power'], 96)
    assert x.shape == (1, 192)
    assert x.dtype == np.float32
    assert list(x[0, :96]) == [i % 5 for i in range(96)]
    assert list(x[0, 96:]) == [1000 + 10 * i for i in range(96)]


def test_vector_respects_channel_count():
    x = _frame_features(_frame(channel_count=10), ['counts', 'power'], 96)
    assert np.count_nonzero(x[0, 10:96]) == 0
    assert np.count_nonzero(x[0, 106:]) == 0
    assert x[0, 96] == 1000


def test_bundle_round_trips_through_pickle_and_predicts():
    bundle = pickle.loads(pickle.dumps({'pipeline': _fitted(192), 'layout': LAYOUT}))
    pipeline, layout = _check_model_bundle(bundle, 'mem')
    assert layout['features'] == ['counts', 'power']
    pred = pipeline.predict(_frame_features(_frame(), layout['features'], 96))
    assert pred.shape == (1,)
    assert int(pred[0]) in range(4)


def test_bundle_rejects_width_mismatch():
    with pytest.raises(RuntimeError, match='192'):
        _check_model_bundle({'pipeline': _fitted(96), 'layout': LAYOUT}, 'mem')


def test_bundle_rejects_unknown_feature():
    layout = dict(LAYOUT, features=['counts', 'lfp'])
    with pytest.raises(RuntimeError, match='lfp'):
        _check_model_bundle({'pipeline': _fitted(192), 'layout': layout}, 'mem')
