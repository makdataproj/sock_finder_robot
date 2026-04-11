from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sock_finder_robot'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alexander Mak',
    maintainer_email='alexmakdeveloper@gmail.com',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sock_mask_target_trt_node = yahboom_yolov8.sock_mask_target_trt_node:main',
            'target_lock_node = yahboom_yolov8.target_lock_node:main',
            'move_to_sock_mask_ik_node = yahboom_yolov8.move_to_sock_mask_ik_node:main',
            'serial_fsr_node = yahboom_yolov8.serial_fsr_node:main',
            'scan_pickup_sock_ik_node = yahboom_yolov8.scan_pickup_sock_ik_node:main',
            'carry_sock_to_person_node = yahboom_yolov8.carry_sock_to_person_node:main',
            'patrol_node = yahboom_yolov8.patrol_node:main',
            'combined_trt_node = yahboom_yolov8.combined_trt_node:main',
        ],
    },
)