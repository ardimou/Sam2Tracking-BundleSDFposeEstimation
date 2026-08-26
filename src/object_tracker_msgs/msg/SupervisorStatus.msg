# Output of the tracking_supervisor_node - the arbiter of trust.
uint8 IDLE = 0
uint8 GROUNDING = 1
uint8 INITIALIZING = 2
uint8 TRACKING = 3
uint8 UNCERTAIN = 4
uint8 LOST = 5
uint8 REINITIALIZING = 6

std_msgs/Header header
uint8 state
float32 confidence
string failure_reason
