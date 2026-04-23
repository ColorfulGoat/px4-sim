from setuptools import setup

package_name = 'yolo_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'yolo_node = yolo_detection.yolo_node:main',
        ],
    },
)
