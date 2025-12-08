"""
Offline Track Stitching Module

Performs hierarchical track stitching on JSONL outputs from online tracking:
- Stage 1: Within-video stitching using ReID features
- Stage 2: Cross-camera stitching using room constraints
- Gallery matching for identity anchoring (optional)
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


ROOM_PAIRS = {
    "room1": ["016", "019"],
    "room2": ["017", "018"],
}

CAMERA_TO_ROOM = {
    "016": "room1",
    "019": "room1",
    "017": "room2",
    "018": "room2",
}



class VideoTrackStitcher:
    """Class for stitching tracklets within a single camera."""
    




if __name__ == "__main__":
    pass