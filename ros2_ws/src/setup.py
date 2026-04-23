from setuptools import setup

package_name = 'point_cloud_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'pc_node = point_cloud_node.pc_node:main',
        ],
    },
)
