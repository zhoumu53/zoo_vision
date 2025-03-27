uint32 MAX_DETECTION_COUNT=15           # Fix capacity to enable shared-memory zero-copy between processes
uint32 INDIVIDUAL_COUNT=5               # Fix capacity to enable shared-memory zero-copy between processes

zoo_msgs/Header header                  # Header timestamp should be acquisition time of image

float32 scalex_image_from_detection
float32 scaley_image_from_detection

uint32 detection_count

uint32[15] track_ids                    # size=MAX_DETECTION_COUNT
uint32[15] identity_ids                 # size=MAX_DETECTION_COUNT
float32[75] identity_logits             # size=MAX_DETECTION_COUNT*INDIVIDUAL_COUNT

uint32[15] behaviour_ids                 # size=MAX_DETECTION_COUNT
float32[75] behaviour_logits             # size=MAX_DETECTION_COUNT*INDIVIDUAL_COUNT

zoo_msgs/BoundingBox2D[15] bboxes       # size=MAX_DETECTION_COUNT
float32[45] world_positions             # size=MAX_DETECTION_COUNT*3
zoo_msgs/Tensor3b32m masks

zoo_msgs/Timings timings