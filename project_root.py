# This file lives at the root folder of the repo
# It is symlinked into every python folder so that importing it adds the
# repo root to the python path and we can use qualified imports
# e.g.
#   import project_root
#   from scripts.datasets.segmentation_utils import bbox_from_mask

from pathlib import Path


def _get_project_root():
    p = Path.cwd()
    while p.name != "zoo_vision":
        if p == p.parent:
            raise RuntimeError(
                "Could not find a path named zoo_vision in the hierarchy. Cannot determine project root."
            )
        p = p.parent
    return p


# Global constant so we can find files inside of the repo
PROJECT_ROOT = _get_project_root()
DATASETS_ROOT = Path("/home/dherrera/data/")

# Set python path so we can import modules from inside the repo
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
