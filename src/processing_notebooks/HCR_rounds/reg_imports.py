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

sys.path.append("../../")
from registrations import (HCR_confocal_imaging,
                                 verify_rounds)
from bigstream_functions import custom_easifish_registration_pipeline, register_lowres, get_registration_score
# Path for bigstream unless you did pip install
sys.path = [fr"\\nasquatch\data\2p\jonna\Code_Python\Notebooks_Jonna\BigStream\bigstream_v2_andermann"] + sys.path 
sys.path = [fr"C:\Users\jonna\Notebooks_Jonna\BigStream\bigstream_v2_andermann"] + sys.path 
sys.path = [fr'{os.getcwd()}/bigstream_v2_andermann'] + sys.path
sys.path = ["/mnt/nasquatch/data/2p/jonna/Code_Python/Notebooks_Jonna/BigStream/bigstream_v2_andermann"] + sys.path 


# Path for bigstream unless you did pip install
sys.path = ["/mnt/nasquatch/data/2p/jonna/Code_Python/Notebooks_Jonna/BigStream/bigstream_v2_andermann"] + sys.path 

from bigstream.align import feature_point_ransac_affine_align
from bigstream.piecewise_transform import distributed_apply_transform
from bigstream.transform import apply_transform

