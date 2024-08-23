import os
import sys
import zarr
import socket
from pathlib import Path
import hjson
import re
import numpy as np
from tqdm.auto import trange, tqdm

from IPython.display import display, HTML
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imwrite
sys.path.append("../../")
from registrations_funcs import HCR_confocal_imaging, register_lowres, verify_rounds


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
from bigstream.transform import apply_transform
from bigstream.piecewise_transform import distributed_apply_transform
from registrations_funcs import HCR_confocal_imaging, register_lowres, verify_rounds

