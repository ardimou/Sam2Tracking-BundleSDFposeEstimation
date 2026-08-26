"""ASUS Xtion driver with registered depth and the RGB-D adapter."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue

from object_tracker_bringup.launch_helpers import params_argument, params_file


def generate_launch_description():
    namespace = LaunchConfiguration('camera_namespace')
    params = params_file()
    camera = ComposableNodeContainer(
        name='xtion_container', namespace=namespace,
        package='rclcpp_components', executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='openni2_camera',
                plugin='openni2_wrapper::OpenNI2Driver',
                name='driver', namespace=namespace,
                parameters=[{
                    'device_id': ParameterValue(
                        LaunchConfiguration('device_id'), value_type=str),
                    'depth_registration': True,
                    'color_depth_synchronization': True,
                    'use_device_time': True,
                    'color_mode': 'VGA_30Hz',
                    'depth_mode': 'VGA_30Hz',
                    'rgb_frame_id': [namespace, '_rgb_optical_frame'],
                    'depth_frame_id': [namespace, '_rgb_optical_frame'],
                    'ir_frame_id': [namespace, '_ir_optical_frame'],
                }],
                remappings=[('depth/image', 'depth_registered/image_raw')],
            ),
        ], output='screen')
    adapter = Node(
        package='object_tracker_perception', executable='rgbd_adapter_node',
        name='rgbd_adapter_node', output='screen', parameters=[params])
    return LaunchDescription([
        params_argument(),
        DeclareLaunchArgument('camera_namespace', default_value='camera'),
        DeclareLaunchArgument('device_id', default_value='#1'),
        camera, adapter,
    ])
