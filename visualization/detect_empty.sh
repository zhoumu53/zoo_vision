cd ..

video='/mnt/camera_nas/ZAG-ELP-CAM-016/20240905AM/ZAG-ELP-CAM-016-20240905-104719-1725526039568-7.mp4'
video='/mnt/camera_nas/ZAG-ELP-CAM-016/20240905PM/ZAG-ELP-CAM-016-20240905-144719-1725540439722-7.mp4'
video='/mnt/camera_nas/ZAG-ELP-CAM-016/20240905AM/ZAG-ELP-CAM-016-20240905-024719-1725497239539-7.mp4'
video='/mnt/camera_nas/ZAG-ELP-CAM-016/20240905PM/ZAG-ELP-CAM-016-20240905-184718-1725554838999-7.mp4'
video='/mnt/camera_nas/ZAG-ELP-CAM-016/20240905PM/ZAG-ELP-CAM-016-20240905-224718-1725569238475-7.mp4'

python visualization/detect_empty_frames.py \
  --video $video \
  --output reports/ZAG-ELP-CAM-016-20240905-224718-1725569238475-7.csv \
  --yolo-model models/segmentation/yolo/all_v3/weights/best.pt
