import os
import re
import socket
import sys
from pathlib import Path

import hjson
import numpy as np
import zarr
from IPython.display import HTML, display
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imwrite
from tqdm.auto import tqdm, trange

sys.path.append("../../")
from registrations import (HCR_confocal_imaging, register_lowres,
                                 verify_rounds)

# Path for bigstream unless you did pip install
sys.path = [fr"\\nasquatch\data\2p\jonna\Code_Python\Notebooks_Jonna\BigStream\bigstream_github"] + sys.path 
sys.path = [fr"C:\Users\jonna\Notebooks_Jonna\BigStream\bigstream_github"] + sys.path 
sys.path = [fr'{os.getcwd()}/bigstream_github'] + sys.path
sys.path = ["/mnt/nasquatch/data/2p/jonna/Code_Python/Notebooks_Jonna/BigStream/bigstream_github"] + sys.path 


# Path for bigstream unless you did pip install
sys.path = ["/mnt/nasquatch/data/2p/jonna/Code_Python/Notebooks_Jonna/BigStream/bigstream_github"] + sys.path 
sys.path.append("../../")
from bigstream.align import feature_point_ransac_affine_align
from bigstream.application_pipelines import easifish_registration_pipeline
from bigstream.piecewise_transform import distributed_apply_transform
from bigstream.transform import apply_transform
from registrations import (HCR_confocal_imaging, register_lowres,
                                 verify_rounds)
