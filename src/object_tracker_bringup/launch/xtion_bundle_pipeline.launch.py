import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('object_tracker_bringup')
    params = LaunchConfiguration('params_file')
    interval = LaunchConfiguration('launch_interval_sec')

    def node(package, executable, name):
        return Node(
            package=package, executable=executable, name=name,
            output='screen', parameters=[params])

    def delayed(action, step):
        return TimerAction(
            period=PythonExpression([interval, ' * ', str(step)]),
            actions=[action])

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(share, 'config', 'params.yaml')),
        DeclareLaunchArgument('launch_interval_sec', default_value='2.0'),
        node('object_tracker_gui', 'object_tracker_gui', 'object_tracker_gui'),
        delayed(node('object_tracker_perception', 'grounding_node',
                     'grounding_node'), 1),
        delayed(node('object_tracker_perception', 'segmentation_node',
                     'segmentation_node'), 2),
        delayed(node('object_tracker_tracking', 'tracking_supervisor_node',
                     'tracking_supervisor_node'), 3),
        delayed(node('object_tracker_bringup', 'table_frame_node',
                     'table_frame_node'), 4),
        delayed(node('object_tracker_bringup', 'visualization_node',
                     'visualization_node'), 4),
    ])
