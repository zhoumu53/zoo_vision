# This message contains an uncompressed image
# (0, 0) is at top-left corner of image

uint32 height                # image height, that is, number of rows
uint32 width                 # image width, that is, number of columns
uint32 step                  # Full row length in bytes

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new char[256] format, join
# ros-users@lists.ros.org and send an email proposing a new encoding.

string encoding              # Encoding of pixels -- channel meaning, ordering, size
                             # taken from the list of char[256]s in include/sensor_msgs/image_encodings.hpp

uint8[] data                 # actual matrix data, size is (step * rows)
