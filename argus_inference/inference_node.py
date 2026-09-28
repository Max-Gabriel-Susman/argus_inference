# inference_node.py
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
