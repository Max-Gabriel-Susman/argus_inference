"""
The Argus decoder node: one NeuralFrame in, one Twist out, twenty times a second.

Where it sits
-------------

This is the last stage of the Argus Cybernetics stack. Upstream, the codec
in the Arty Z7's fabric computes, for each of 96 channels and each 50 ms
bin, a threshold-crossing count and a mean-square spike-band power; the
firmware sends one frame per bin over UDP; ``argus_sensors`` receives it
and bridges it onto ``/argus/sensors/neural_telemetry`` as a
:class:`argus_core.msg.NeuralFrame`. This node subscribes to that topic,
classifies each frame into one of four reach intents, and publishes a
:class:`geometry_msgs.msg.Twist` on ``/cmd_vel``, the velocity command a
ROS robot base listens for::

    NeuralFrame (channels[96], power[96], sample, t)
        -> feature vector -> StandardScaler -> LDA -> intent in {0, 1, 2, 3}
        -> Twist (linear.x, angular.z) on /cmd_vel

Nothing consumes ``/cmd_vel`` yet; the loop is open by design at this
version.

Two model paths
---------------

Which one is active is logged at startup on a line that begins
``model path:``.

``ARGUS_MODEL_PATH`` set
    The demo path. The file is a pickle written by
    ``argus_sim/tools/decode_test.py --save-model``: a dict with a fitted
    scikit-learn ``pipeline`` and a ``layout`` that describes its input,
    ``{'features': ['counts', 'power'], 'channels': 96, 'bin_len': 1500,
    'mult': 3.5}``. :func:`_check_model_bundle` refuses a bundle whose
    feature names are unknown or whose pipeline width is not
    ``len(features) * channels``, and :func:`_frame_features` builds the
    vector in layout order, ``[counts..., power...]``. With counts plus
    power at 3.5 sigma that model scores 53.5 % in 5-fold cross-validation
    over the full session; ``decode_test.py`` holds the comparison with
    the lab's spike-sorted units. Unpickling runs code: load only pickles
    you made.

``ARGUS_MODEL_PATH`` unset
    The original path, kept for the counts-only comparison. The node reads
    the Sabes lab session given by ``ARGUS_DATASET_PATH`` (MATLAB v7.3, so
    HDF5), bins the spike times of unit ``ARGUS_UNIT_INDEX`` on each
    channel (1 by default, the first sorted unit; 0 is the unsorted hash)
    into ``ARGUS_BIN_S`` bins, fits the same StandardScaler and LDA
    pipeline on an 80/20 stratified split, and logs the held-out accuracy
    (about 44 %). The feature vector is then counts only,
    ``ARGUS_MAX_CHANNELS`` wide.

Intents and the command
-----------------------

Labels come from the session's behaviour: at each bin centre, the angle
from the cursor to the reach target, quadrantised. The saved model was
trained on the same four classes. Each intent maps to one fixed command
(:func:`_intent_to_twist`); up on the screen is forward, left and right
are turns, down is reverse:

======  ================  =========  ===========
intent  reach direction   linear.x   angular.z
======  ================  =========  ===========
0       right (+x)        0.15 m/s   -0.70 rad/s
1       up (+y)           0.25 m/s   0.00 rad/s
2       left (-x)         0.15 m/s   0.70 rad/s
3       down (-y)         -0.15 m/s  0.00 rad/s
======  ================  =========  ===========

This is a demonstration mapping, four fixed commands. A continuous
decoder, a regressor on cursor velocity instead of a classifier on
direction, would drop into the same node and publish the same message
type.

Environment and parameters
--------------------------

``ARGUS_MODEL_PATH``
    Saved model bundle; selects the demo path. The launch exports it from
    ``model:=``.
``ARGUS_DATASET_PATH``
    The session ``.mat``; required when ``ARGUS_MODEL_PATH`` is unset.
``ARGUS_BIN_S``
    Bin length in seconds for training at startup. Default ``0.05``, the
    fabric's bin.
``ARGUS_UNIT_INDEX``
    Which unit's spike times to bin per channel when training at startup.
    Default ``1``.
``ARGUS_MAX_CHANNELS``
    Width of the counts-only vector. Default ``96``.

ROS parameters: ``input_topic`` (default ``/argus/sensors/neural_telemetry``)
and ``cmd_topic`` (default ``/cmd_vel``), set by ``argus_bringup``'s
``config/argus.yaml``. The subscription uses the sensor-data QoS profile
(best effort, keep last 5), so a late host drops frames rather than
queueing them; the publisher is reliable with depth 10.

Running it
----------

The stack brings the node up through the launch::

    ros2 launch argus_bringup argus.launch.py model:=$HOME/argus_model.pkl

or on its own::

    ARGUS_MODEL_PATH=$HOME/argus_model.pkl ros2 run argus_inference inference_node

Every twentieth frame, once a second at the fabric's rate, the node logs
``sample=... t=... intent=... -> vx=... wz=...``. An empty frame
(``channel_count == 0``) is warned about and skipped. The feature-vector
and bundle contracts are pinned by ``test/test_model_path.py``; the
offline validation behind the accuracy figures is
``argus_sim/tools/decode_test.py``.
"""
from dataclasses import dataclass
import math
import os
import pickle
from typing import Optional, Sequence, Tuple

from argus_core.msg import NeuralFrame
from geometry_msgs.msg import Twist
import h5py
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Dataset:
    t: np.ndarray
    cursor_xy: np.ndarray
    target_xy: np.ndarray
    spikes_ref: np.ndarray


def _load_mat_v73(path: str) -> Dataset:
    with h5py.File(path, "r") as f:
        t = np.array(f["/t"]).squeeze().astype(np.float64)
        cursor = np.array(f["/cursor_pos"]).T.astype(np.float64)
        target = np.array(f["/target_pos"]).T.astype(np.float64)
        spikes_ref = np.array(f["/spikes"])

    return Dataset(t=t, cursor_xy=cursor, target_xy=target, spikes_ref=spikes_ref)


def _read_spike_times(f: h5py.File, ref: h5py.Reference) -> np.ndarray:
    if ref == 0:
        return np.empty((0,), dtype=np.float64)

    dset = f[ref]
    arr = np.array(dset).squeeze()
    return arr.astype(np.float64).reshape(-1)


def _build_features_and_labels(
    path: str,
    bin_s: float,
    unit_index: int,
    max_channels: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = _load_mat_v73(path)
    t = ds.t
    t0 = float(t[0])
    t1 = float(t[-1])

    edges = np.arange(t0, t1 + bin_s, bin_s, dtype=np.float64)
    bin_times = edges[:-1] + 0.5 * bin_s
    n_bins = bin_times.shape[0]

    n_units, n_ch = ds.spikes_ref.shape
    _ = n_units
    if max_channels is None:
        max_channels = n_ch
    ch_count = min(max_channels, n_ch)

    X = np.zeros((n_bins, ch_count), dtype=np.float32)

    with h5py.File(path, "r") as f:
        for ch in range(ch_count):
            ref = ds.spikes_ref[unit_index, ch]
            st = _read_spike_times(f, ref)
            if st.size == 0:
                continue
            counts, _ = np.histogram(st, bins=edges)
            X[:, ch] = counts.astype(np.float32)

    idx = np.searchsorted(t, bin_times, side="left")
    idx = np.clip(idx, 0, t.shape[0] - 1)

    vec = ds.target_xy[idx] - ds.cursor_xy[idx]
    dx = vec[:, 0]
    dy = vec[:, 1]
    ang = np.arctan2(dy, dx)

    y = np.zeros((n_bins,), dtype=np.int64)
    y[(ang >= math.pi / 4) & (ang < 3 * math.pi / 4)] = 1
    y[(ang >= 3 * math.pi / 4) | (ang < -3 * math.pi / 4)] = 2
    y[(ang >= -3 * math.pi / 4) & (ang < -math.pi / 4)] = 3

    return X, y, bin_times


_FRAME_FIELDS = {"counts": "channels", "power": "power"}


def _load_model_bundle(path: str):
    """
    Load a decode_test.py --save-model pickle: (pipeline, layout).

    Only load pickles you made; unpickling runs code.
    """
    with open(path, "rb") as f:
        return _check_model_bundle(pickle.load(f), path)


def _check_model_bundle(bundle, source: str):
    pipeline = bundle["pipeline"]
    layout = bundle["layout"]
    features = list(layout["features"])
    unknown = [name for name in features if name not in _FRAME_FIELDS]
    if not features or unknown:
        raise RuntimeError(
            f"{source}: layout features {features} not all in {sorted(_FRAME_FIELDS)}"
        )
    channels = int(layout["channels"])
    expected = len(features) * channels
    n_in = getattr(pipeline, "n_features_in_", expected)
    if n_in != expected:
        raise RuntimeError(
            f"{source}: pipeline takes {n_in} features, layout describes {expected}"
        )
    return pipeline, layout


def _frame_features(msg, features: Sequence[str], channels: int) -> np.ndarray:
    """
    Build the feature vector, [counts..., power...] in layout order.

    Channels beyond channel_count are left at zero, as the .mat path does.
    """
    x = np.zeros((1, len(features) * channels), dtype=np.float32)
    for k, name in enumerate(features):
        values = getattr(msg, _FRAME_FIELDS[name])
        usable = min(int(msg.channel_count), channels, len(values))
        for i in range(usable):
            x[0, k * channels + i] = float(values[i])
    return x


def _intent_to_twist(intent: int) -> Twist:
    msg = Twist()

    if intent == 1:
        msg.linear.x = 0.25
        msg.angular.z = 0.0
    elif intent == 3:
        msg.linear.x = -0.15
        msg.angular.z = 0.0
    elif intent == 2:
        msg.linear.x = 0.15
        msg.angular.z = 0.7
    else:
        msg.linear.x = 0.15
        msg.angular.z = -0.7

    return msg


class InferenceNode(Node):

    def __init__(self) -> None:
        super().__init__("argus_inference")

        model_path = os.environ.get("ARGUS_MODEL_PATH")
        self._layout = None

        self._bin_s = float(os.environ.get("ARGUS_BIN_S", "0.05"))
        self._unit_index = int(os.environ.get("ARGUS_UNIT_INDEX", "1"))
        self._max_channels = int(os.environ.get("ARGUS_MAX_CHANNELS", "96"))

        self._input_topic = (
            self.declare_parameter(
                "input_topic", "/argus/sensors/neural_telemetry"
            ).value
        )
        self._cmd_topic = (
            self.declare_parameter(
                "cmd_topic", "/cmd_vel"
            ).value
        )

        self._pub = self.create_publisher(Twist, self._cmd_topic, 10)

        if model_path:
            self._load_saved_model(os.path.expanduser(model_path))
        else:
            self._train_on_mat()

        self._rx_count = 0

        self._sub = self.create_subscription(
            NeuralFrame,
            self._input_topic,
            self._neural_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"argus_inference online: subscribing to '{self._input_topic}' "
            f"and publishing '{self._cmd_topic}'"
        )

    def _load_saved_model(self, path: str) -> None:
        self._model, self._layout = _load_model_bundle(path)
        self.get_logger().info(
            f"model path: ARGUS_MODEL_PATH={path} (saved model; "
            f"features={self._layout['features']} channels={self._layout['channels']} "
            f"mult={self._layout.get('mult')} bin_len={self._layout.get('bin_len')})"
        )

    def _train_on_mat(self) -> None:
        dataset_path = os.environ.get("ARGUS_DATASET_PATH")
        if not dataset_path:
            raise RuntimeError(
                "Neither ARGUS_MODEL_PATH nor ARGUS_DATASET_PATH is set.\n"
                "Example:\n"
                "  ARGUS_DATASET_PATH=$HOME/Documents/datasets/indy_loco/indy_20161005_06.mat "
                "ros2 run argus_inference inference_node"
            )
        self._dataset_path = os.path.expanduser(dataset_path)

        self.get_logger().info(
            "model path: training on the .mat at startup (counts only; "
            "set ARGUS_MODEL_PATH to use a saved model)"
        )
        self.get_logger().info(f"dataset: {self._dataset_path}")
        self.get_logger().info(
            f"bin_s={self._bin_s} unit_index={self._unit_index} "
            f"max_channels={self._max_channels}"
        )

        X, y, _ = _build_features_and_labels(
            path=self._dataset_path,
            bin_s=self._bin_s,
            unit_index=self._unit_index,
            max_channels=self._max_channels,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=7, stratify=y
        )

        self._model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LinearDiscriminantAnalysis(),
        )
        self._model.fit(X_train, y_train)
        acc = float(self._model.score(X_test, y_test))
        self.get_logger().info(f"offline 4-way intent accuracy: {acc:.3f}")

    def _frame_to_features(self, msg: NeuralFrame) -> np.ndarray:
        if self._layout is not None:
            return _frame_features(
                msg, self._layout["features"], int(self._layout["channels"])
            )

        x = np.zeros((1, self._max_channels), dtype=np.float32)

        usable = min(int(msg.channel_count), self._max_channels, len(msg.channels))
        for i in range(usable):
            x[0, i] = float(msg.channels[i])

        return x

    def _neural_callback(self, msg: NeuralFrame) -> None:
        if msg.channel_count == 0:
            self.get_logger().warning("Received empty neural frame")
            return

        x = self._frame_to_features(msg)
        pred = int(self._model.predict(x)[0])

        cmd = _intent_to_twist(pred)
        self._pub.publish(cmd)

        if self._rx_count % 20 == 0:
            self.get_logger().info(
                f"sample={msg.sample} t={msg.t:.3f} intent={pred} -> "
                f"vx={cmd.linear.x:.2f} wz={cmd.angular.z:.2f}"
            )

        self._rx_count += 1


def main() -> None:
    rclpy.init()
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
