# Structured output of the language-guided grounding node.
# Published on-demand: at track start, on target change, and on recovery.
std_msgs/Header header
# Exact RGB frame on which the boxes were predicted. Grounding runs
# asynchronously, so timestamp-only caches may evict this frame before SAM2.
sensor_msgs/Image source_image
string query
uint32 object_id             # primary/highest-confidence instance
string object_description
float32[4] bounding_box     # [x_min, y_min, x_max, y_max] in pixel coordinates
float32 confidence
GroundingCandidate[] alternative_candidates
bool success
string failure_reason
