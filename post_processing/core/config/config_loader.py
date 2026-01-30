"""Configuration loader for post-processing pipeline."""
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    reid_config: Path
    reid_checkpoint: Path
    reid_gallery_path: Optional[Path]
    behavior_model_path: Path


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    device: str
    batch_size: int
    sample_rate: float
    overwrite_behavior: bool
    overwrite_reid: bool


@dataclass
class DataConfig:
    """Data paths configuration."""
    record_root: Path
    output_dir: Path


@dataclass
class CameraConfig:
    """Camera configuration."""
    ids: List[str]
    height: int
    width: int


@dataclass
class TimeWindowConfig:
    """Time window configuration."""
    start_time: str
    end_time: str


@dataclass
class StitchingConfig:
    """Stitching parameters configuration."""
    run_stitching: bool
    max_gap_frames: int
    local_sim_th: float
    gallery_sim_th: float
    head_k: int
    tail_k: int
    gallery_k: int
    w_local: float
    w_gallery: float
    num_identities: int


@dataclass
class CrossCameraConfig:
    """Cross-camera settings."""
    enabled: bool
    run_behavior_matching: bool


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    models: ModelConfig
    processing: ProcessingConfig
    data: DataConfig
    cameras: CameraConfig
    time_window: TimeWindowConfig
    stitching: StitchingConfig
    cross_camera: CrossCameraConfig
    log_level: str
    
    def update_from_dict(self, updates: dict) -> None:
        """Update config values from a dictionary.
        
        Args:
            updates: Dictionary with keys in dot notation (e.g., 'processing.batch_size')
                    and values to set
        
        Example:
            config.update_from_dict({
                'processing.batch_size': 128,
                'cameras.ids': ['016', '017'],
                'data.record_root': '/new/path'
            })
        """
        for key_path, value in updates.items():
            keys = key_path.split('.')
            if len(keys) < 2:
                continue
                
            section = keys[0]
            attr = keys[1]
            
            # Get the section object
            if not hasattr(self, section):
                continue
            
            section_obj = getattr(self, section)
            
            # Update the attribute
            if hasattr(section_obj, attr):
                # Convert to appropriate type
                original = getattr(section_obj, attr)
                try:
                    if isinstance(original, Path):
                        converted = Path(value)
                    elif isinstance(original, bool):
                        converted = value if isinstance(value, bool) else str(value).lower() in ('true', '1', 'yes')
                    elif isinstance(original, (int, float, str, list)):
                        converted = type(original)(value) if not isinstance(value, type(original)) else value
                    else:
                        converted = value
                    
                    setattr(section_obj, attr, converted)
                except (ValueError, TypeError):
                    pass  # Skip invalid conversions


def load_config(config_path: Path | str) -> PipelineConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        PipelineConfig object with all settings
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Parse model config
    models = ModelConfig(
        reid_config=Path(data['models']['reid']['config']),
        reid_checkpoint=Path(data['models']['reid']['checkpoint']),
        reid_gallery_path=Path(data['models']['reid']['gallery_path']) if data['models']['reid']['gallery_path'] else None,
        behavior_model_path=Path(data['models']['behavior']['model_path']),
    )
    
    # Parse processing config
    processing = ProcessingConfig(
        device=data['processing']['device'],
        batch_size=data['processing']['batch_size'],
        sample_rate=data['processing']['sample_rate'],
        overwrite_behavior=data['processing']['overwrite_behavior'],
        overwrite_reid=data['processing']['overwrite_reid'],
    )
    
    # Parse data config
    data_config = DataConfig(
        record_root=Path(data['directories']['record_root']),
        output_dir=Path(data['directories']['output_dir']),
    )
    
    # Parse camera config
    cameras = CameraConfig(
        ids=data['cameras']['ids'],
        height=data['cameras']['height'],
        width=data['cameras']['width'],
    )
    
    # Parse time window
    time_window = TimeWindowConfig(
        start_time=data['time_window']['start_time'],
        end_time=data['time_window']['end_time'],
    )
    
    # Parse stitching config
    stitching = StitchingConfig(
        run_stitching=data['stitching']['run_stitching'],
        max_gap_frames=data['stitching']['max_gap_frames'],
        local_sim_th=data['stitching']['local_sim_th'],
        gallery_sim_th=data['stitching']['gallery_sim_th'],
        head_k=data['stitching']['head_k'],
        tail_k=data['stitching']['tail_k'],
        gallery_k=data['stitching']['gallery_k'],
        w_local=data['stitching']['w_local'],
        w_gallery=data['stitching']['w_gallery'],
        num_identities=data['stitching']['num_identities'],
    )
    
    # Parse cross-camera config
    cross_camera = CrossCameraConfig(
        enabled=data['cross_camera']['enabled'],
        run_behavior_matching=data['cross_camera']['run_behavior_matching'],
    )
    
    return PipelineConfig(
        models=models,
        processing=processing,
        data=data_config,
        cameras=cameras,
        time_window=time_window,
        stitching=stitching,
        cross_camera=cross_camera,
        log_level=data['logging']['level'],
    )


def update_config_from_args(config: PipelineConfig, **kwargs) -> None:
    """Update config from command-line arguments.
    
    Args:
        config: PipelineConfig object to update
        **kwargs: Keyword arguments with values to override
        
    Example:
        update_config_from_args(config, 
                               record_root='/new/path',
                               batch_size=128,
                               camera_ids=['016', '017'])
    """
    updates = {}
    
    # Map common argument names to config paths
    arg_to_config = {
        'record_root': 'data.record_root',
        'output_dir': 'data.output_dir',
        'batch_size': 'processing.batch_size',
        'sample_rate': 'processing.sample_rate',
        'device': 'processing.device',
        'camera_ids': 'cameras.ids',
        'cam_ids': 'cameras.ids',
        'start_time': 'time_window.start_time',
        'end_time': 'time_window.end_time',
        'overwrite_behavior': 'processing.overwrite_behavior',
        'overwrite_reid': 'processing.overwrite_reid',
    }
    
    for arg_name, value in kwargs.items():
        if value is None:
            continue
            
        # Use mapping or assume dot notation
        config_path = arg_to_config.get(arg_name, arg_name)
        updates[config_path] = value
    
    if updates:
        config.update_from_dict(updates)


if __name__ == "__main__":
    # Test config loading
    config_path = Path(__file__).parent / "configs.yaml"
    config = load_config(config_path)
    
    print("Configuration loaded successfully!")
    print(f"Record root: {config.data.record_root}")
    print(f"Output dir: {config.data.output_dir}")
    print(f"ReID checkpoint: {config.models.reid_checkpoint}")
    print(f"Cameras: {config.cameras.ids}")
    print(f"Batch size: {config.processing.batch_size}")
    print(f"Device: {config.processing.device}")
    
    # Test updating config
    print("\nTesting config update...")
    update_config_from_args(config, 
                           batch_size=128,
                           camera_ids=['016', '017'],
                           record_root='/tmp/test')
    
    print(f"Updated batch size: {config.processing.batch_size}")
    print(f"Updated cameras: {config.cameras.ids}")
    print(f"Updated record root: {config.data.record_root}")