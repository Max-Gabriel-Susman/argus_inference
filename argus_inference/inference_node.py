import math
from dataclasses import dataclass
import os
from typing import Optional, Tuple

import h5py
import numpy as np
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Dataset:
    t: np.ndarray           # (T,)
    cursor_xy: np.ndarray   # (T, 2)
    target_xy: np.ndarray   # (T, 2)
    spikes_ref: np.ndarray  # (units, channels) of object refs


def _load_mat_v73(path: str) -> Dataset:
    with h5py.File(path, "r") as f:
        t = np.array(f["/t"]).squeeze().astype(np.float64)  # (T,)
        cursor = np.array(f["/cursor_pos"]).T.astype(np.float64)  # (T,2)
        target = np.array(f["/target_pos"]).T.astype(np.float64)  # (T,2)
        spikes_ref = np.array(f["/spikes"])  # (5,96) dtype=object (refs)

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

    n_units, n_ch = ds.spikes_ref.shape  # (5,96)
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

    # 4-way: 0=RIGHT, 1=UP, 2=LEFT, 3=DOWN
    y = np.zeros((n_bins,), dtype=np.int64)
    y[(ang >= math.pi / 4) & (ang < 3 * math.pi / 4)] = 1
    y[(ang >= 3 * math.pi / 4) | (ang < -3 * math.pi / 4)] = 2
    y[(ang >= -3 * math.pi / 4) & (ang < -math.pi / 4)] = 3

    return X, y, bin_times


def _intent_to_twist(intent: int) -> Twist:
    msg = Twist()

    # Simple mapping:
    # 0 RIGHT  -> turn right while moving forward a bit
    # 1 UP     -> forward
    # 2 LEFT   -> turn left while moving forward a bit
    # 3 DOWN   -> reverse
    if intent == 1:
        msg.linear.x = 0.25
        msg.angular.z = 0.0
    elif intent == 3:
        msg.linear.x = -0.15
        msg.angular.z = 0.0
    elif intent == 2:
        msg.linear.x = 0.15
        msg.angular.z = 0.7
    else:  # intent == 0
        msg.linear.x = 0.15
        msg.angular.z = -0.7

    return msg


class InferenceNode(Node):
    """ROS 2 proof-of-concept node: decode reaching intent and publish /cmd_vel."""

    def __init__(self) -> None:
        super().__init__("argus_inference")

        dataset_path = os.environ.get("ARGUS_DATASET_PATH")
        if not dataset_path:
            raise RuntimeError(
                "ARGUS_DATASET_PATH is not set.\n"
                "Example:\n"
                "  ARGUS_DATASET_PATH=$HOME/Documents/datasets/indy_loco/indy_20161005_06.mat "
                "ros2 run argus_inference inference_node"
            )
        self._dataset_path = os.path.expanduser(dataset_path)

        self._bin_s = float(os.environ.get("ARGUS_BIN_S", "0.05"))
        self._unit_index = int(os.environ.get("ARGUS_UNIT_INDEX", "1"))
        self._max_channels = int(os.environ.get("ARGUS_MAX_CHANNELS", "96"))

        self._pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.get_logger().info(f"dataset: {self._dataset_path}")
        self.get_logger().info(
            f"bin_s={self._bin_s} unit_index={self._unit_index} "
            f"max_channels={self._max_channels}"
        )

        X, y, bin_times = _build_features_and_labels(
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

        # Replay state
        self._X_replay = X
        self._bin_times = bin_times
        self._i = 0

        period_s = self._bin_s
        self._timer = self.create_timer(period_s, self._tick)

        self.get_logger().info("argus_inference online (publishing /cmd_vel)")

    def _tick(self) -> None:
        if self._i >= self._X_replay.shape[0]:
            self.get_logger().info("replay finished; publishing stop and shutting down")
            stop = Twist()
            self._pub.publish(stop)
            rclpy.shutdown()
            return

        x = self._X_replay[self._i: self._i + 1]
        pred = int(self._model.predict(x)[0])

        cmd = _intent_to_twist(pred)
        self._pub.publish(cmd)

        # log occasionally to avoid spamming
        if self._i % 20 == 0:
            bt = float(self._bin_times[self._i])
            self.get_logger().info(
                f"t={bt:.2f}s intent={pred} -> "
                f"vx={cmd.linear.x:.2f} wz={cmd.angular.z:.2f}"
            )

        self._i += 1


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
