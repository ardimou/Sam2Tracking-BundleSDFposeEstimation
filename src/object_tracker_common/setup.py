from setuptools import find_packages, setup

package_name = 'object_tracker_common'
setup(
    name=package_name, version='0.1.0', packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='you', maintainer_email='you@example.com',
    description='Shared topic names and QoS profiles for object tracker packages.',
    license='Apache-2.0',
)
