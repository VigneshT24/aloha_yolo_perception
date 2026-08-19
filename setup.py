from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'aloha_yolo_pickup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Config files (yaml + rviz)
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vignesh Thondikulam',
    maintainer_email='vignesh.tho2006@gmail.com',
    description='YOLO object detection and sensor fusion for Mobile ALOHA',
    license='TODO: License',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'yolo_detection_node = aloha_yolo_pickup.yolo_detection_node:main',
            'sensor_fusion_node = aloha_yolo_pickup.sensor_fusion_node:main'
        ],
    },
)