Instructions for intern
========================

# How to initialize the ZooVision environment
Start terminal
```
cd git/zoo_vision/                      # Change directory to where the code is
source ./activate_ros.sh                # Run the script that sets environment and selects python interpreter
```
All the following sections assume you have initialized the environment.

# How to view the code
```
code .vscode/zoo_vision.code-workspace
```

# Downloading updates
```
git pull								# This downloads the updates from the web
ninja -C build/RelWithDebInfo			# This rebuilds the app
```
# Downloading recordings
The NAS can be accessed from `/mnt/camera_nas/`.
The cameras we care about are `ZAG-ELP-CAM-016` to `ZAG-ELP-CAM-019`.
Copy the videos to our database directory: `/media/dherrera/ElephantsWD/elephants/videos/zali_era`
Reconstruct the video database to include the new videos with:
```
python scripts/datasets/make_video_time_db.py -i /media/dherrera/ElephantsWD/elephants/videos
```
After updating the database you can pass the replay datetime to ZooVision and it will automatically find the video (see section "Running ZooVision").

# Labelling videos for detection
```
python labelling/video_labeller/__main__.py --video_dir <dir_with_videos> --label_dir <dir_to_store_labelling> [--behaviour]
```
For example, the video database that was originally labelled is at `/media/dherrera/ElephantsWD/elephants/videos/identity_days`. 
The `--behaviour` parameter selects what labels to use. Without it the labels are the elephant names, with it the sleep postures.

# Running ZooVision
To start ZooVision and collect tracks from live camera:
```
build/RelWithDebInfo/src/zoo_vision/zoo_vision -c /live_stream=true -c /record_tracks=true -c /db/enabled=true
```
To start ZooVision and collect tracks from videos:
```
build/RelWithDebInfo/src/zoo_vision/zoo_vision -c /replay_time=\"2025-03-18T14:40:00\" -c /record_tracks=true
```
Tracks are stored in `/media/dherrera/ElephantsWD/elephants/improve/tracks`.
Once the tracks have been recorded they will have too many images. Clean up duplicates and remove small tracks with:
```
python scripts/datasets/cleaning/compress_tracks.py -i /media/dherrera/ElephantsWD/elephants/improve/tracks
```

# Linux tips
`Ctrl-R` in the terminal searches for past commands. Very useful to not type everything from scratch.

# Systemd commands
These commands allow us to write programs in the background without needing to be logged in the machine.
```
systemd-run --user -u <svc_name> <cmd>
systemctl status --user <svc_name>
systemctl stop --user <svc_name>

# Show logs in console
journalctl --user -u <svc_name>

# Dump logs to text file
journalctl --user --no-tail -u <svc_name> > log.txt
```
An example to run the tracker in the background:
```
systemd-run --user -u zoo_vision_svc bash /home/dherrera/git/zoo_vision/build/RelWithDebInfo/run_zoo_vision.sh
systemctl status --user zoo_vision_svc
journalctl --user -u zoo_vision_svc
```

