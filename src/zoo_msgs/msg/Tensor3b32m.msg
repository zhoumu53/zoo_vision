# This message a 3D tensor with u8 data type.

uint32 DATA_MAX_SIZE=12582912                   # Fix capacity to enable shared-memory zero-copy between processes
                                                # size = 12*1024*1024
uint32[3] sizes
                                                # Data is stored in row-major order and contiguous.
uint8[12582912] data                            # Element (i,j,k) = data[i*sizes[1]*sizes[2] + j*sizes[2] + k]
                                                # Always assert that the data fits this message size.


