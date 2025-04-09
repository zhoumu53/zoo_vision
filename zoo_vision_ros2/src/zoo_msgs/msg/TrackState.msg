zoo_msgs/Header header                  # Header timestamp should be acquisition time of image

uint32 track_id
zoo_msgs/Image8m keyframe_mosaic

uint32 selected_identity
float32[5] identity_votes               # size=INDIVIDUAL_COUNT
