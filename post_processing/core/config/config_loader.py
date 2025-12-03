"""
Configuration Loader

Utilities for loading and managing pipeline configuration from YAML file.
"""
from __future__ import annotations

import os
from pathlib import Path
from string import Template
from typing import Any, Dict, Optional

import yaml


class Config:
    """Pipeline configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file. If None, uses default config.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "configs.yaml"
        
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self._raw_config = yaml.safe_load(f)
        
        self._config = self._resolve_variables(self._raw_config)
    
    def _resolve_variables(self, obj: Any, context: Optional[Dict] = None) -> Any:
        """Recursively resolve ${var} references in config."""
        if context is None:
            context = {}
        
        if isinstance(obj, dict):
            resolved = {}
            for key, value in obj.items():
                resolved[key] = self._resolve_variables(value, context)
                if key not in context:
                    context[key] = resolved[key]
            return resolved
        
        elif isinstance(obj, list):
            return [self._resolve_variables(item, context) for item in obj]
        
        elif isinstance(obj, str):
            if '${' in obj:
                template = Template(obj)
                try:
                    return template.safe_substitute(**self._flatten_dict(self._raw_config))
                except (KeyError, ValueError):
                    return obj
            return obj
        
        return obj
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for template substitution."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-separated key path."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Dict-like access to config."""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None
    
    @property
    def directories(self) -> Dict:
        """Get all directory paths."""
        return self._config.get('directories', {})
    
    @property
    def models(self) -> Dict:
        """Get all model paths."""
        return self._config.get('models', {})
    
    @property
    def devices(self) -> Dict:
        """Get device configuration."""
        return self._config.get('devices', {})
    
    @property
    def cameras(self) -> Dict:
        """Get camera configuration."""
        return self._config.get('cameras', {})
    
    @property
    def identities(self) -> Dict:
        """Get identity mapping."""
        return self._config.get('identities', {})
    
    def format_path(self, template: str, **kwargs) -> str:
        """Format path template with provided variables."""
        t = Template(template)
        return t.safe_substitute(**kwargs)
    
    def get_video_path(self, cam_id: str, date: str, ampm: str) -> Path:
        """Get video directory path for camera, date, and AM/PM."""
        template = self.get('video_patterns.directory_format')
        raw_video_dir = self.get('directories.raw_video_dir')
        
        path = self.format_path(
            template,
            raw_video_dir=raw_video_dir,
            cam_id=cam_id,
            date=date,
            ampm=ampm
        )
        return Path(path)
    
    def get_output_dir(self, stage: str, date: str, hour: int) -> Path:
        """Get output directory for processing stage."""
        key_map = {
            'online': 'directories.online_tracking_output',
            'offline': 'directories.offline_stitching_output',
            'postproc': 'directories.post_processing_output',
            'behavior': 'directories.behavior_analysis_output',
        }
        
        key = key_map.get(stage)
        if not key:
            raise ValueError(f"Unknown stage: {stage}")
        
        template = self.get(key)
        output_base = self.get('directories.output_base_dir')
        
        path = self.format_path(
            template,
            output_base_dir=output_base,
            date=date,
            hour=f"{hour:02d}"
        )
        return Path(path)
    
    def to_dict(self) -> Dict:
        """Export configuration as dictionary."""
        return self._config.copy()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    return Config(config_path)


if __name__ == "__main__":
    # Test config loading
    config = load_config()
    
    print("Configuration loaded successfully!")
    print(f"Project root: {config.get('directories.project_root')}")
    print(f"Raw video dir: {config.get('directories.raw_video_dir')}")
    print(f"ReID checkpoint: {config.get('models.reid.checkpoint')}")
    print(f"Cameras: {config.get('cameras.ids')}")
    print(f"Identities: {config.get('identities.id_to_name')}")
    
    # Test path formatting
    video_path = config.get('models')
    print(f"Video path example: {video_path}")