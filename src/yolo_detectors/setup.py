from setuptools import find_packages, setup

package_name = 'yolo_detectors'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Charlotte Di Loreto',
    maintainer_email='charlotte@example.com',
    description='YOLO26 object detection and pose estimation with Orbbec Femto Bolt',
    license='MIT',
    entry_points={
        'console_scripts': [
            'alien_detection = yolo_detectors.alien_detection_node:main',
            'pose_detection  = yolo_detectors.pose_detection_node:main',
            'camera_driver = yolo_detectors.camera_driver_node:main',
            'alien_yolo = yolo_detectors.alien_yolo_node:main',
            'pose_yolo = yolo_detectors.pose_yolo_node:main',
            'test_subscriber = yolo_detectors.test_subscriber_node:main',
            'test_yolo = yolo_detectors.test_yolo_node:main'
        ],
    },
)
