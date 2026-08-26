from setuptools import find_packages, setup

package_name = 'object_tracker_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Xtion RGB-D adaptation, GroundingDINO, and SAM 2 nodes.',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'rgbd_adapter_node = object_tracker_perception.rgbd_adapter_node:main',
        'grounding_node = object_tracker_perception.grounding_node:main',
        'segmentation_node = object_tracker_perception.segmentation_node:main',
    ]},
)
