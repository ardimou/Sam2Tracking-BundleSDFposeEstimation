from glob import glob
from setuptools import find_packages, setup

package_name = 'object_tracker_bringup'
setup(
    name=package_name, version='0.1.0', packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='you', maintainer_email='you@example.com',
    description='Xtion and BundleSDF pipeline launch and visualization.',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'visualization_node = object_tracker_bringup.visualization_node:main',
        'table_frame_node = object_tracker_bringup.table_frame_node:main',
    ]},
)
