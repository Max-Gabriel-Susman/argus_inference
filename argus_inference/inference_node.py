import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class InferenceNode(Node):
    """ROS 2 motor cortex intent prediction proof of concept node."""

    def __init__(self):
        super().__init__('argus_inference')
        self._pub = self.create_publisher(String, '/argus/effectors/cmd', 10)
        self._counter = 0
        self.create_timer(0.5, self._tick)
        self.get_logger().info('argus_inference online')

    def _tick(self):
        msg = String()
        msg.data = f'poc_cmd_{self._counter}'
        self._counter += 1
        self._pub.publish(msg)
        self.get_logger().info(f'published: {msg.data}')


def main():
    rclpy.init()
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
