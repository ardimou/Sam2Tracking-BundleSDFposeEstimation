from setuptools import find_packages, setup

package_name = 'object_tracker_tracking'
setup(
    name=package_name, version='0.1.0', packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], tests_require=['pytest'], zip_safe=True,
    maintainer='you', maintainer_email='you@example.com',
    description='Multi-instance BundleTrack pose estimation and supervision.',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'bundlesdf_tracking_node = object_tracker_tracking.bundlesdf_tracking_node:main',
        'tracking_supervisor_node = object_tracker_tracking.tracking_supervisor_node:main',
    ]},
)
