cd ..


PROJECT_ROOT=/media/mu/zoo_vision/training/PoseGuidedReID

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT


video_path=/home/mu/Desktop/gt_videos/train/ZAG-ELP-CAM-018-20250830-025815-1756515495749-7.mp4
opposite_video_path=/mnt/camera_nas/ZAG-ELP-CAM-017/20250830AM/ZAG-ELP-CAM-017-20250830-025750-1756515470201-7.mp4
max_frames=1000

# python3 visualization/video_reid.py \
#   --video "$video_path" \
#   --output runs/test.mp4 \
#   --yolo-model models/segmentation/yolo/all_v3/weights/best.pt \
#   --class-names models/segmentation/yolo/class_names.txt \
#   --reid-config training/PoseGuidedReID/configs/elephant_resnet.yml \
#   --reid-checkpoint training/PoseGuidedReID/logs/elephant_resnet/lr01_bs32/net_best.pth \
#   --gallery training/PoseGuidedReID/logs/elephant_resnet/lr01_bs32/pred_features/train_iid/pytorch_result_e.npz \
#   --device cuda --gallery-device cpu --top-k 3 --max-frames $max_frames



# python3 visualization/video_id_classification.py \
#   --video "$video_path" \
#   --output runs/test.mp4 \
#   --yolo-model models/segmentation/yolo/all_v3/weights/best.pt \
#   --class-names models/segmentation/yolo/class_names.txt \
#   --id-checkpoint models/identity/vit/v4/config.ptc \
#   --device cuda --max-frames $max_frames

# python3 visualization/video_tracks_reid.py \
#   --video "$video_path" \
#   --output runs/reid_track_vis_matches.mp4 \
#   --yolo-model models/segmentation/yolo/all_v3/weights/best.pt \
#   --class-names models/segmentation/yolo/class_names.txt \
#   --reid-config training/PoseGuidedReID/configs/elephant_resnet.yml \
#   --reid-checkpoint training/PoseGuidedReID/logs/elephant_resnet/lr01_bs32/net_best.pth \
#   --gallery training/PoseGuidedReID/logs/elephant_resnet/lr01_bs32/pred_features/train_iid/pytorch_result_e.npz \
#   --tracker-config bytetrack.yaml \
#   --min-similarity 0.5 \
#   --device cuda --gallery-device cpu --max-frames $max_frames


# python3 visualization/comparison.py \
#   --video "$opposite_video_path" \
#   --output runs/comparison_opposite_${max_frames}.mp4 \
#   --reid-output runs/reid_opposite_${max_frames}.mp4 \
#   --track-output runs/track_reid_opposite_${max_frames}.mp4 \
#   --id-output runs/id_opposite_${max_frames}.mp4 \
#   --class-names models/segmentation/yolo/class_names.txt \
#   --yolo-model models/segmentation/yolo/all_v3/weights/best.pt \
#   --reid-config training/PoseGuidedReID/configs/elephant_resnet.yml \
#   --reid-checkpoint training/PoseGuidedReID/logs/elephant_resnet/lr01_bs32/net_best.pth \
#   --gallery training/PoseGuidedReID/logs/elephant_resnet/lr01_bs32/pred_features/train_iid/pytorch_result_e.npz \
#   --id-checkpoint ../models/identity/vit/v4/config.ptc \
#   --tracker-config bytetrack.yaml \
#   --titles ReID "ReID+Track" "ID Classifier" \
#   --max-frames $max_frames \



python3 visualization/run_multi_camera.py \
  --video "$opposite_video_path" \
  --track-outdir /home/mu/Desktop/comparison_videos/improved \
  --class-names models/segmentation/yolo/class_names.txt \
  --yolo-model models/segmentation/yolo/all_v3/weights/best.pt \
  --reid-config training/PoseGuidedReID/configs/elephant_resnet.yml \
  --reid-checkpoint training/PoseGuidedReID/logs/elephant_resnet/lr01_bs32/net_best.pth \
  --gallery training/PoseGuidedReID/logs/elephant_resnet/lr01_bs32/pred_features/train_iid/pytorch_result_e.npz \
  --id-checkpoint ../models/identity/vit/v4/config.ptc \
  --tracker-config bytetrack.yaml \
  --max-frames $max_frames \