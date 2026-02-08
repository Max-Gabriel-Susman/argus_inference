import rclpy
from rclpy.node import Node


class InferenceNode(Node):
  def __init__(self):
    super().__init__("argus_inference")
    self.get_logger().info("argus_inference online")

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

if __name__ == "__main__": 
  main()
