# Reported alongside /object/mask. Mirrors the mask lifecycle.
uint8 INITIALIZING = 0
uint8 TRACKING = 1
uint8 UNCERTAIN = 2
uint8 LOST = 3
uint8 SWITCHED = 4    # identity check suggests mask jumped to a different object

std_msgs/Header header
uint8 state
float32 confidence
