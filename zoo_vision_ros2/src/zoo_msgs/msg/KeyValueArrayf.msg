uint32 MAX_ITEMS=15       # Fix capacity to enable shared-memory zero-copy between processes

uint32 item_count
zoo_msgs/String[15] keys  # size=MAX_ITEMS
float32[15] values        # size=MAX_ITEMS
