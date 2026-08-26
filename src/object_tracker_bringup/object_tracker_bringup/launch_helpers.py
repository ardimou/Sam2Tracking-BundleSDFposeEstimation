import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def params_argument():
    default = os.path.join(
        get_package_share_directory('object_tracker_bringup'), 'config', 'params.yaml')
    return DeclareLaunchArgument(
        'params_file', default_value=default,
        description='ROS parameter YAML used by the selected components')


def params_file():
    return LaunchConfiguration('params_file')
