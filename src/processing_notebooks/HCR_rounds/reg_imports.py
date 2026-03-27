import os
import re
import socket
import sys
from pathlib import Path

import hjson
import numpy as np
import zarr
import shutil
from IPython.display import HTML, display
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imwrite
from tqdm.auto import tqdm, trange

# Add src directory to path BEFORE importing from registrations
# This ensures registrations_utils can be found as an absolute import
_src_path = str(Path(__file__).resolve().parent.parent.parent)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from registrations import (HCR_confocal_imaging,
                                 verify_rounds)
from bigstream_functions import custom_easifish_registration_pipeline, register_lowres, get_registration_score

# BigStream import: tries pip-installed version first, then falls back to common locations.
# To use a custom BigStream path, set the BIGSTREAM_PATH environment variable, e.g.:
#   export BIGSTREAM_PATH=/home/user/BigStream/bigstream_v2_andermann
try:
    from bigstream.align import feature_point_ransac_affine_align
except ImportError:
    _bigstream_candidates = [
        os.environ.get('BIGSTREAM_PATH', ''),                       # user-configured
        str(Path.cwd() / 'bigstream_v2_andermann'),                 # local to notebook
        str(Path(_src_path).parent / 'BigStream' / 'bigstream_v2_andermann'),  # sibling to ezfish repo
    ]
    _found = False
    for _bp in _bigstream_candidates:
        if _bp and Path(_bp).is_dir():
            sys.path.insert(0, _bp)
            _found = True
            break
    if not _found:
        raise ImportError(
            "BigStream not found. Install it via pip, or set the BIGSTREAM_PATH environment variable "
            "to point to your bigstream_v2_andermann directory.\n"
            "  e.g.:  export BIGSTREAM_PATH=/path/to/BigStream/bigstream_v2_andermann"
        )
    from bigstream.align import feature_point_ransac_affine_align

from bigstream.piecewise_transform import distributed_apply_transform
from bigstream.transform import apply_transform

