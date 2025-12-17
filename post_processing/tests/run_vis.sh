


# python visualize_stitched_tracks.py \
#   --stitched_json /media/dherrera/ElephantsWD/elephants/test/tracks/zag_elp_cam_016/2025-11-29/stitched_tracklets_cam016_153000_235959.json \
#   --frame_width 1920 --frame_height 1080 \
#   --out_dir /media/mu/zoo_vision/post_processing/scripts/old/global_overlay_empty \
#   --write_every 5 \
#   --save_frames \
#   --camera_id 016 \
#   --path_contains 2025-11-29



for date in 2025-12-04 2025-11-29  2025-11-30  2025-12-01  2025-12-03 
do

    for camera_id in 016 017 018 019
    do
        for start_end in '153000_235959' '000000_150000'
        
        do
        jsonfile=/media/dherrera/ElephantsWD/elephants/test/tracks/zag_elp_cam_$camera_id/${date}/stitched_tracklets_cam${camera_id}_${start_end}.json
        # echo "Processing file: $jsonfile"
        ## check if file exists
        if [ ! -f $jsonfile ]; then
            echo "File not found: $jsonfile"
            continue
        fi

        python visualize_overlay_on_raw_videos.py \
            --stitched_json $jsonfile \
            --raw_root /mnt/camera_nas \
            --out_dir /media/mu/zoo_vision/post_processing/scripts/old/raw2 \
            --save_frames

        done
    done
done



# python visualize_overlay_on_raw_videos.py \
#     --stitched_json /media/dherrera/ElephantsWD/elephants/test/tracks/zag_elp_cam_016/2025-11-29/stitched_tracklets_cam016_153000_235959.json \
#     --raw_root /mnt/camera_nas \
#     --out_dir /media/mu/zoo_vision/post_processing/scripts/old/raw2 \
#     --save_frames
