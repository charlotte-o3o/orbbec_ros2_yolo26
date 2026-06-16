import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/charlottediloreto/ros2_orbbec_ws/install/yolo_detectors'
