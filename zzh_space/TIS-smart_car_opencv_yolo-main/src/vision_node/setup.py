from setuptools import find_packages, setup

package_name = 'vision_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhangjingxuan',
    maintainer_email='1607782513@qq.com',
    description='i love u',
    license='MIT',  # 使用 SPDX 格式
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
        'vision_node = vision_node.vision_node:main',
        'vision_node_2 = vision_node.vision_node_2:main',
        'vision_node_3 = vision_node.vision_node_3:main'
        ],
    },
)

