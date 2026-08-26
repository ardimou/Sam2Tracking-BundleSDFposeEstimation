# Raw, per-frame output of a pose tracking node.
# This is *not* the trusted pose - see SupervisorStatus / /object/pose for that.
std_msgs/Header header
geometry_msgs/PoseStamped t_camera_object
bool tracking_success
float32 tracking_quality        # 0-1 backend confidence/inlier proxy
float32 processing_time_ms
string reconstruction_status
