cmake_minimum_required(VERSION 3.8)
project(object_tracker_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GroundingCandidate.msg"
  "msg/GroundingResult.msg"
  "msg/MaskState.msg"
  "msg/InstanceMask.msg"
  "msg/InstanceTrackerStatus.msg"
  "msg/InstanceTablePose.msg"
  "msg/InstanceSupervisorStatus.msg"
  "msg/TrackerStatus.msg"
  "msg/SupervisorStatus.msg"
  "srv/ExportMesh.srv"
  "srv/SaveModel.srv"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_package()
