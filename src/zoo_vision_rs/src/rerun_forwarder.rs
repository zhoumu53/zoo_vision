use crate::zoo_config::ClassifierClassInfo;
use time;

use super::zoo_config::ZooConfig;
use anyhow::{Error, Result};
use hex_color::HexColor;
use image;
use nalgebra::{Matrix3, Matrix4, Vector3};
use ndarray::prelude::*;
use re_ws_comms::RerunServerPort;
use rerun::{
    demo_util::grid,
    external::{glam, ndarray, re_log_types},
    RecordingStream,
};
use std::{
    collections::HashMap,
    io::Cursor,
    iter::zip,
    path::{Path, PathBuf},
    rc::Rc,
};

type RosString = zoo_msgs::msg::rmw::String;

fn nanosec_from_ros(stamp: &builtin_interfaces::msg::rmw::Time) -> i64 {
    1e9 as i64 * stamp.sec as i64 + stamp.nanosec as i64
}

fn string_from_ros(ros_str: &RosString) -> String {
    let view = unsafe { std::str::from_utf8_unchecked(ros_str.data.as_slice()) };
    view.to_string()
}

const LOW_RES_WIDTH: u32 = 320;
const RERUN_APP_ID: &str = "zoo_vision";

fn transform3d_from_2d(t2d: &Matrix3<f32>) -> Matrix4<f32> {
    let mut t3d: Matrix4<f32> = nalgebra::zero();
    for r in [0, 1] {
        for c in [0, 1] {
            t3d[(r, c)] = t2d[(r, c)];
        }
    }
    for r in [0, 1] {
        t3d[(r, 3)] = t2d[(r, 2)];
    }
    t3d[(2, 2)] = 1.0;
    t3d[(3, 3)] = 1.0;
    return t3d;
}

fn cast_ros_key_value_arrayi64(
    ros_array: &zoo_msgs::msg::rmw::KeyValueArrayi64,
) -> Vec<(String, i64)> {
    let count = ros_array.item_count as usize;
    let keys = &ros_array.keys[0..count];
    let values = &ros_array.values[0..count];
    zip(
        keys.iter().map(|x| string_from_ros(x)),
        values.iter().map(|x| *x),
    )
    .collect()
}
fn cast_ros_key_value_arrayf(ros_array: &zoo_msgs::msg::rmw::KeyValueArrayf) -> Vec<(String, f32)> {
    let count = ros_array.item_count as usize;
    let keys = &ros_array.keys[0..count];
    let values = &ros_array.values[0..count];
    zip(
        keys.iter().map(|x| string_from_ros(x)),
        values.iter().map(|x| *x),
    )
    .collect()
}

fn rerun_from_hex(color: &HexColor) -> rerun::Color {
    rerun::Color::from_unmultiplied_rgba(color.r, color.g, color.b, color.a)
}

fn start_rerun_file_recording(recording_filename: &str) -> Result<RecordingStream, Error> {
    let dt_fmt = time::format_description::parse(recording_filename)?;

    let local_offset = time::UtcOffset::current_local_offset()?;
    let now = time::UtcDateTime::now().to_offset(local_offset);
    let path = now.format(&dt_fmt)?;
    println!("Saving rerun stream to {}", path);

    let stream_builder = rerun::RecordingStreamBuilder::new(RERUN_APP_ID);
    return Ok(stream_builder.save(path)?);
}

fn log_static_components(
    recording: &RecordingStream,
    data_path: &Path,
    config: &ZooConfig,
    key_colors: Option<&HashMap<String, rerun::Color>>,
) -> Result<(), Error> {
    // Log the blueprint
    recording.log_file_from_path(data_path.join("zoo_vision.rbl"), None, true)?;

    // Load floor plan
    let t_map_from_world2 =
        Matrix3::<f32>::from_row_slice(config.map.t_map_from_world2.as_flattened());
    let t_map_from_world = transform3d_from_2d(&t_map_from_world2);

    let map_path = data_path.join(&config.map.image);
    println!("Map filename={}", map_path.to_str().unwrap());
    let world_image_rr = rerun::EncodedImage::from_file(map_path)?;
    recording.log_static("world/map", &world_image_rr)?;

    // Map projection
    // {
    //     // let f = [t_map_from_world[(0, 0)], t_map_from_world[(1, 1)]];
    //     // let p = [t_map_from_world[(0, 2)], t_map_from_world[(1, 2)]];
    //     let resolution = [4904.0, 7663.0];
    //     let f = [-1.0, -1.0];
    //     recording.log_static(
    //         "/world/map",
    //         &rerun::Pinhole::from_focal_length_and_resolution(f, resolution)
    //             .with_image_plane_distance(1.0),
    //     )?;
    // }
    let t_world_from_map = t_map_from_world.qr().try_inverse().unwrap();
    let r = t_world_from_map.fixed_view::<3, 3>(0, 0).clone_owned();
    let t = t_world_from_map.fixed_view::<3, 1>(0, 3).clone_owned();
    // t[2] = -1.0;

    recording.log_static(
        "world/map",
        &rerun::Transform3D::from_mat3x3(r.data.0).with_translation(t.data.0[0]),
    )?;

    // Go through config cameras
    for (_, (camera_name, camera_config)) in config.cameras.iter().enumerate() {
        // Log pinhole in map view
        let resolution = [
            camera_config.intrinsics.width as f32,
            camera_config.intrinsics.height as f32,
        ];
        let f = [
            camera_config.intrinsics.k[0][0],
            camera_config.intrinsics.k[1][1],
        ];
        let p = [
            camera_config.intrinsics.k[0][2],
            camera_config.intrinsics.k[1][2],
        ];
        recording.log_static(
            format!("/world/{}", camera_name),
            &rerun::Pinhole::from_focal_length_and_resolution(f, resolution)
                .with_principal_point(p)
                .with_image_plane_distance(3.0),
        )?;

        let r_world_from_camera = Matrix3::<f32>::from_row_slice(
            camera_config.t_world_from_camera.rotation.as_flattened(),
        );
        let t_camera_in_world = Vector3::<f32>::from(camera_config.t_world_from_camera.translation);
        recording.log_static(
            format!("/world/{}", camera_name),
            &rerun::Transform3D::from_mat3x3(r_world_from_camera.data.0)
                .with_translation(t_camera_in_world.data.0[0]),
        )?;
    }

    // Log an annotation context to assign a label and color to each class
    let mut identity_ctx = vec![(
        0u16,
        "Background",
        rerun::Rgba32::from_unmultiplied_rgba(0, 0, 0, 0),
    )];
    for (name, individual) in config.individuals.iter() {
        let color = individual.color;
        identity_ctx.push((
            individual.id as u16,
            name,
            rerun::Rgba32::from_unmultiplied_rgba(color.r, color.g, color.b, 128),
        ));
    }
    for (camera_name, _) in config.cameras.iter() {
        recording.log_static(
            format!("/cameras/{}/detections", camera_name),
            &rerun::AnnotationContext::new([(
                0,
                "Background",
                rerun::Rgba32::from_unmultiplied_rgba(0, 0, 0, 0),
            )]),
        )?;
        recording.log_static(
            format!("/cameras/{}/identities", camera_name),
            &rerun::AnnotationContext::new(identity_ctx.clone()),
        )?;
    }
    recording.log_static(
        format!("/world/detections"),
        &rerun::AnnotationContext::new(identity_ctx),
    )?;

    if key_colors.is_some() {
        for (rr_path, rr_color) in key_colors.unwrap() {
            recording.log_static(
                rr_path.as_str(),
                &rerun::SeriesPoint::new().with_color(rr_color.clone()),
            )?;
        }
    }
    Ok(())
}

pub struct RerunForwarder {
    config: ZooConfig,
    data_path: PathBuf,

    recording_mutex: std::sync::Mutex<Rc<rerun::RecordingStream>>,

    first_ros_time_ns: Option<i64>,
    // camera_indices: HashMap<String, usize>,
    identity_from_id: HashMap<u32, (String, ClassifierClassInfo)>,
    behaviour_from_id: HashMap<u32, (String, ClassifierClassInfo)>,

    camera_short_from_name: HashMap<String, String>,
    camera_ui_scale_name: HashMap<String, f32>,

    key_colors: HashMap<String, rerun::Color>,
}

impl RerunForwarder {
    pub fn new(data_path: &Path, config_json: &str) -> Result<Self, Error> {
        // Load config
        let config: ZooConfig = serde_json::from_str(config_json).expect("Config json not valid");

        // Begin rerun stream
        let recording = if config.rerun_config.save_to_disk {
            start_rerun_file_recording(&config.rerun_config.recording_filename)?
        } else {
            let stream_builder = rerun::RecordingStreamBuilder::new(RERUN_APP_ID);
            let ws_port = 9877u16;
            println!("Broadcasting rerun stream. Visualize with 'rerun ws://localhost:{} --memory-limit 1GB'", ws_port);
            stream_builder.serve_web(
                "0.0.0.0",
                Default::default(),
                RerunServerPort(ws_port),
                rerun::MemoryLimit::from_bytes(1024 * 1024 * 1024),
                false,
            )?
        };

        log_static_components(&recording, data_path, &config, None)?;

        const ASCII_A: u8 = 'a' as u8;
        let identity_from_id = HashMap::from_iter(
            config
                .individuals
                .iter()
                .map(|(name, class_info)| (class_info.id, (name.clone(), class_info.clone()))),
        );
        let behaviour_from_id = HashMap::from_iter(
            config
                .behaviours
                .iter()
                .map(|(name, class_info)| (class_info.id, (name.clone(), class_info.clone()))),
        );
        let camera_short_from_name =
            HashMap::from_iter(config.cameras.iter().enumerate().map(|(i, (name, _))| {
                (
                    name.to_owned(),
                    format!("{}", (ASCII_A + (i as u8)) as char),
                )
            }));
        let camera_ui_scale_name = HashMap::from_iter(
            config
                .cameras
                .iter()
                .map(|(name, _)| (name.to_owned(), 1f32)),
        );

        Ok(Self {
            config,
            data_path: data_path.to_path_buf(),
            recording_mutex: std::sync::Mutex::new(Rc::new(recording)),
            first_ros_time_ns: None,
            // camera_indices: HashMap::new(),
            identity_from_id,
            behaviour_from_id,
            camera_short_from_name,
            camera_ui_scale_name,
            key_colors: HashMap::new(),
        })
    }

    pub fn get_recording(&mut self) -> Rc<RecordingStream> {
        // Lock the mutex to clone the Rc, this protects from creation of a new recording stream while we are logging
        // but allows multithreaded log() calls.
        let lock_guard = self.recording_mutex.lock().unwrap();
        Rc::clone(&lock_guard)
    }

    pub fn test_me(&mut self, frame_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let points = grid(glam::Vec3::splat(-10.0), glam::Vec3::splat(10.0), 10);
        let colors = grid(glam::Vec3::ZERO, glam::Vec3::splat(255.0), 10)
            .map(|v| rerun::Color::from_rgb(v.x as u8, v.y as u8, v.z as u8));
        let recording = self.get_recording();
        recording.log(
            "my_points",
            &rerun::Points3D::new(points)
                .with_colors(colors)
                .with_radii([0.5]),
        )?;
        println!("Test from forwarder, frame_id={}", frame_id);
        Ok(())
    }

    pub fn reset_recording_if_necessary(&mut self) -> () {
        if !self.config.rerun_config.save_to_disk {
            return ();
        }

        // Recording is multi-threaded but creation must be guarded with a mutex so two threads
        // don't attempt to start a new recording simultaneously
        let mut lock_guard = self.recording_mutex.lock().unwrap();

        let recording_time = lock_guard.store_info().unwrap().started;
        let recording_duration = re_log_types::Time::now() - recording_time;
        let max_duration = self.config.rerun_config.max_seconds_per_recording as f32;
        if recording_duration.as_secs_f32() > max_duration {
            *lock_guard = Rc::new(
                start_rerun_file_recording(&self.config.rerun_config.recording_filename)
                    .expect("Could not start new recording"),
            );
            log_static_components(
                &lock_guard,
                &self.data_path,
                &self.config,
                Some(&self.key_colors),
            )
            .expect("Error logging static components to new recording");
        }
    }

    pub fn register_points_key(&mut self, rr_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if self.key_colors.contains_key(rr_path) {
            return Ok(());
        }

        let color = hex_color::HexColor::random_rgb();
        let alpha = 100u8;
        let rr_color = rerun::Color::from_unmultiplied_rgba(color.r, color.g, color.b, alpha);

        self.key_colors.insert(rr_path.to_string(), rr_color);

        self.get_recording()
            .log_static(rr_path, &rerun::SeriesPoint::new().with_color(rr_color))?;
        Ok(())
    }

    pub fn image_callback(
        &mut self,
        camera: &str,
        _channel: &str,
        msg: &zoo_msgs::msg::rmw::Image12m,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: it is a bit excessive to check for a new recording on every frame, but the check is light
        self.reset_recording_if_necessary();

        let msg_data_slice = unsafe {
            std::slice::from_raw_parts(msg.data.as_ptr(), (msg.height * msg.width * 3) as usize)
        };
        let image = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(
            msg.width,
            msg.height,
            msg_data_slice,
        )
        .unwrap();

        // Compress image
        let mut jpg_writer = Cursor::new(Vec::new());
        if self.config.rerun_config.low_res {
            // Resize before compressing
            let aspect = image.width() as f32 / image.height() as f32;
            let low_res_height = (LOW_RES_WIDTH as f32 / aspect) as u32;
            let low_res_image = image::imageops::resize(
                &image,
                LOW_RES_WIDTH,
                low_res_height,
                image::imageops::Gaussian,
            );
            low_res_image.write_to(&mut jpg_writer, image::ImageFormat::Jpeg)?;
            // Remember scale
            let ui_scale = LOW_RES_WIDTH as f32 / image.width() as f32;
            *self.camera_ui_scale_name.get_mut(camera).unwrap() = ui_scale;
        } else {
            image.write_to(&mut jpg_writer, image::ImageFormat::Jpeg)?;
        }

        let image_jpg_data = jpg_writer.into_inner();
        let rr_image =
            rerun::EncodedImage::from_file_contents(image_jpg_data).with_media_type("image/jpeg");

        let time_ns = nanosec_from_ros(&msg.header.stamp);
        let recording = self.get_recording();
        recording.set_time_nanos("ros_time", time_ns);

        recording.log(
            format!("/cameras/{}/fullres", camera),
            &rr_image.with_draw_order(-1.0),
        )?;

        // Clear out detections
        // self.recording.set_time_nanos("ros_time", time_ns - 1);

        // let camera_name = "input_camera";
        // let image_detections_ent = format!("{}/detections", camera_name);
        // self.recording
        //     .log(image_detections_ent, &rerun::Clear::recursive())?;
        // let world_detections_ent = format!("world/{}/detections", camera_name);
        // self.recording
        //     .log(world_detections_ent, &rerun::Clear::recursive())?;
        // println!("Test from forwarder, image id={}", unsafe {
        //     std::str::from_utf8_unchecked(msg.header.frame_id.data.as_slice())
        // });
        Ok(())
    }

    pub fn detection_callback(
        &mut self,
        camera: &str,
        _channel: &str,
        msg: &zoo_msgs::msg::rmw::Detection,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ros_time_ns = nanosec_from_ros(&msg.header.stamp);
        const DROP_SAMPLE_DURATION: i64 = 2 * 1e9 as i64;
        if DROP_SAMPLE_DURATION > 0 {
            match self.first_ros_time_ns {
                Some(t) => {
                    if ros_time_ns < t {
                        return Ok(());
                    }
                }
                None => {
                    self.first_ros_time_ns = Some(ros_time_ns + DROP_SAMPLE_DURATION);
                    return Ok(());
                }
            }
        }

        let recording = self.get_recording();

        // Set rerun time
        recording.set_time_nanos("ros_time", ros_time_ns as i64);

        recording.log(
            format!("/cameras/{}/fullres", camera),
            &rerun::Transform3D::from_mat3x3([
                1.0 / msg.scalex_image_from_detection / self.camera_ui_scale_name[camera],
                0.0,
                0.0,
                0.0,
                1.0 / msg.scaley_image_from_detection / self.camera_ui_scale_name[camera],
                0.0,
                0.0,
                0.0,
                1.0,
            ]),
        )?;

        let detection_count = msg.detection_count as usize;
        // let track_ids: Vec<u16> = (0..detection_count)
        //     .map(|x| msg.track_ids[x] as u16)
        //     .collect();
        let identity_ids: Vec<u16> = (0..detection_count)
            .map(|x| msg.identity_ids[x] as u16)
            .collect();
        let labels: Vec<String> = (0..detection_count)
            .map(|x| {
                let id = msg.identity_ids[x];
                let behaviour_id = msg.behaviour_ids[x];
                if self.identity_from_id.contains_key(&id) == false {
                    print!("About to panic with identity id=={}", id);
                }
                format!(
                    "T{}{}-{}-{}",
                    self.camera_short_from_name[camera],
                    msg.track_ids[x],
                    self.identity_from_id[&id].0.as_str(),
                    self.behaviour_from_id[&behaviour_id].0.as_str(),
                )
            })
            .collect();
        // Map message data to image array
        assert!(msg.detection_count == msg.masks.sizes[0]);
        let mask_height = msg.masks.sizes[1] as usize;
        let mask_width = msg.masks.sizes[2] as usize;
        let masks: ArrayBase<ndarray::ViewRepr<&u8>, Ix3> = unsafe {
            ArrayView::from_shape_ptr(
                (detection_count, mask_height, mask_width),
                msg.masks.data.as_ptr(),
            )
        };

        let world_positions: ArrayBase<ndarray::ViewRepr<&f32>, Ix2> = unsafe {
            ArrayView::from_shape_ptr((detection_count, 3), msg.world_positions.as_ptr())
        };

        // Log bounding boxes
        let bbox_centers = (0..detection_count).map(|x| msg.bboxes[x].center);
        let bbox_half_sizes = (0..detection_count).map(|x| msg.bboxes[x].half_size);

        let rerun_boxes =
            rerun::Boxes2D::from_centers_and_half_sizes(bbox_centers, bbox_half_sizes);

        // Log boxes with track ids as labels
        // self.recording.log(
        //     format!("/cameras/{}/detections/boxes", camera),
        //     &rerun_boxes.clone().with_class_ids(track_ids.clone()),
        // )?;

        recording.log(
            format!("/cameras/{}/identities/boxes", camera),
            &rerun_boxes
                .with_class_ids(identity_ids.clone())
                .with_labels(labels)
                .with_show_labels(true),
        )?;

        // Log position in world
        let world_points_rr =
            rerun::Points2D::new(world_positions.axis_iter(Axis(0)).map(|x| (x[0], x[1])));
        recording.log(
            format!("/world/detections/{}/positions", camera),
            &world_points_rr
                .with_class_ids(identity_ids)
                .with_radii([1.0]),
        )?;

        // Log detection masks
        // const THRESHOLD: u8 = (0.8 * 255.0) as u8;
        const THRESHOLD: u8 = 1 as u8;
        let mut image_classes = ndarray::Array2::<u8>::zeros((mask_height, mask_width).f());
        for idx in 0..detection_count {
            let value = msg.track_ids[idx] as u8;
            let mask_i = masks.slice(s![idx, .., ..]);
            for (p, m) in image_classes.iter_mut().zip(mask_i.iter()) {
                if *m >= THRESHOLD {
                    *p = value;
                }
            }
        }
        let rr_image = rerun::SegmentationImage::try_from(image_classes)?;
        recording.log(
            format!("/cameras/{}/detections/masks", camera),
            &rr_image.with_draw_order(1.0).with_opacity(0.35),
        )?;

        // Log processing times
        for (key, value_hz) in cast_ros_key_value_arrayf(&msg.timings.items_hz) {
            let rr_path = format!("/processing_times/hz/{}", key);
            self.register_points_key(rr_path.as_str())?;
            recording.log(rr_path, &rerun::Scalar::new(value_hz as f64))?;
        }
        for (key, value_ns) in cast_ros_key_value_arrayi64(&msg.timings.items_ns) {
            let rr_path = format!("/processing_times/msec/{}", key);
            self.register_points_key(rr_path.as_str())?;

            let value_time = std::time::Duration::from_nanos(value_ns as u64);
            recording.log(
                rr_path,
                &rerun::Scalar::new(value_time.as_secs_f64() * 1000.0),
            )?;
        }

        recording.flush_async();

        Ok(())
    }

    pub fn track_state_callback(
        &mut self,
        camera: &str,
        _channel: &str,
        msg: &zoo_msgs::msg::rmw::TrackState,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ros_time_ns = nanosec_from_ros(&msg.header.stamp);
        let recording = self.get_recording();

        // Set rerun time
        recording.set_time_nanos("ros_time", ros_time_ns as i64);

        // Root path for the track
        let track_name = format!("T{}{}", self.camera_short_from_name[camera], msg.track_id);
        let root_path = format!("/tracks/{}", track_name);

        // Log keyframe image
        {
            let keyframe_path = format!("{}/keyframes/k{}", root_path, msg.new_keyframe_index);

            // Log mosaic transform
            const MOSAIC_ROW_COUNT: u32 = 4;
            const IMAGE_SIZE: u32 = 224;
            let row = msg.new_keyframe_index % MOSAIC_ROW_COUNT;
            let col = msg.new_keyframe_index / MOSAIC_ROW_COUNT;
            recording.log(
                keyframe_path.as_str(),
                &rerun::Transform3D::from_translation([
                    (col * IMAGE_SIZE) as f32,
                    (row * IMAGE_SIZE) as f32,
                    0.0 as f32,
                ]),
            )?;

            // Log keyframe image
            let msg_data_slice = msg.new_keyframe.data.as_slice();
            let image = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(
                msg.new_keyframe.width,
                msg.new_keyframe.height,
                msg_data_slice,
            )
            .unwrap();

            assert!(
                (msg.new_keyframe.width * msg.new_keyframe.height * 3) as usize
                    <= msg_data_slice.len(),
                "width={}, height={}, len={}",
                msg.new_keyframe.width,
                msg.new_keyframe.height,
                msg_data_slice.len()
            );
            // Compress image
            let mut jpg_data = Cursor::new(Vec::new());
            image.write_to(&mut jpg_data, image::ImageFormat::Jpeg)?;

            let rr_image = rerun::EncodedImage::from_file_contents(jpg_data.into_inner())
                .with_media_type("image/jpeg");

            recording.log(keyframe_path, &rr_image.with_draw_order(-1.0))?;
        }

        // Log votes
        {
            for (id_valid, probs) in msg.identity_probs.iter().enumerate() {
                let id = id_valid as u32 + 1;
                let (identity_name, identity_info) = &self.identity_from_id[&id];
                // let logits = all_logits.slice(s![idx, ..]);
                let rr_path = format!("{}/votes/id_{}", root_path, identity_name);

                recording.log(
                    rr_path.as_str(),
                    &rerun::SeriesLine::new().with_color(rerun_from_hex(&identity_info.color)),
                )?;
                recording.log(rr_path, &rerun::Scalar::new(*probs as f64))?;
            }
        }

        Ok(())
    }
}
