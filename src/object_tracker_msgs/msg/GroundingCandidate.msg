# A single candidate grounding produced by the VLM / open-vocabulary detector.
uint32 object_id
string description
float32[4] bounding_box     # [x_min, y_min, x_max, y_max] in pixel coordinates
float32 confidence
