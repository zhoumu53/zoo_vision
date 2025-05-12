zoo_msgs/Header header                  # Header timestamp should be acquisition time of image

uint32 track_id

uint32 new_keyframe_index
zoo_msgs/Image1m new_keyframe

uint32 selected_identity
float32[5] identity_probs               # size=INDIVIDUAL_COUNT
