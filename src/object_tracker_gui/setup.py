from setuptools import find_packages, setup

package_name = 'object_tracker_gui'
setup(
    name=package_name, version='0.1.0', packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='you', maintainer_email='you@example.com',
    description='Qt operator GUI for the object-tracking pipeline.',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'object_tracker_gui = object_tracker_gui.gui_node:main',
    ]},
)
