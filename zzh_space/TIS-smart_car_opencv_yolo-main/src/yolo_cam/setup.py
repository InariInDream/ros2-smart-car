from setuptools import setup
import os
from glob import glob

package_name = 'yolo_cam'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zjx',
    maintainer_email='1607782513@qq.com',
    description='YOLO camera detection package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = yolo_cam.camera_node:main',
            'yolo_node = yolo_cam.yolo_node:main',
        ],
    },
)
