import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from bigstream.align import alignment_pipeline
from bigstream.piecewise_align import distributed_piecewise_alignment_pipeline
from bigstream.piecewise_transform import distributed_apply_transform
from bigstream.transform import apply_transform
from ClusterWrap import cluster as cluster_constructor
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imwrite
from bigstream.align import feature_point_ransac_affine_align
from bigstream.application_pipelines import easifish_registration_pipeline
from bigstream.transform import apply_transform
from bigstream.piecewise_transform import distributed_apply_transform

def custom_easifish_registration_pipeline(
    fix_lowres,
    fix_highres,
    mov_lowres,
    mov_highres,
    fix_lowres_spacing,
    fix_highres_spacing,
    mov_lowres_spacing,
    mov_highres_spacing,
    blocksize,
    write_directory,
    global_ransac_kwargs={},
    global_affine_kwargs={},
    local_ransac_kwargs={},
    local_deform_kwargs={},
    cluster_kwargs={},
    cluster=None,
    only_lowres=False,
    fname='',
    no_deform=False,
    no_global_affine=False,
    overwrite_lowres=False,
    c = {'n_workers': 10, 'threads_per_worker':2},
):
    """
    The easifish registration pipeline.
    Runs 4 registration steps: (1) global affine using ransac on feature points
    extracted with a blob detector, (2) global affine refinement using gradient
    descent of an image matching metric (a function of all image voxels),
    (3) local affine refinement using ransac on feature points again, and
    (4) local deformable refinement using gradient descent of an image
    matching metric and a cubic b-spline parameterized deformation.

    All four steps can be totally customized through their keyword arguments.
    This function assumes steps (1) and (2) can be done in memory on low
    resolution data and steps (3) and (4) must be done in distributed memory
    on high resolution data. Thus, steps (3) and (4) are run on overlapping
    blocks in parallel across distributed resources.

    Parameters
    ----------
    fix_lowres : ndarray
        the fixed image, at lower resolution

    fix_highres : zarr array
        the fixed image, at higher resolution

    mov_lowres : ndarray
        the moving image, at lower resolution

    mov_highres : zarr array
        the moving image, at higher resolution

    fix_lowres_spacing : 1d array
        the spacing in physical units (e.g. mm or um) between
        voxels of the fix_lowres image

    fix_highres_spacing : 1d array
        the spacing in physical units (e.g. mm or um) between
        voxels of the fix_highres image

    mov_lowres_spacing : 1d array
        the spacing in physical units (e.g. mm or um) between
        voxels of the mov_lowres image

    mov_highres_spacing : 1d array
        the spacing in physical units (e.g. mm or um) between
        voxels of the mov_highres image

    blocksize : iterable
        the shape of blocks in voxels

    write_directory : str
        a folder on disk where outputs will be written
        this pipeline will create the following files:
            affine.mat : the global affine transform
            affine.npy : affine.mat applied to mov_lowres
            deform.zarr : the local transform (a vector field)
            deformed.zarr : [affine.mat, deform.zarr] applied to mov_highres

    global_ransac_kwargs : dict
        Any arguments you would like to pass to the global instance of
        bigstream.align.feature_point_ransac_affine_align. See the
        docstring for that function for valid parameters.
        default : {'alignment_spacing':np.min(fix_lowres_spacing)*4,
                   'blob_sizes':[int(round(np.min(fix_lowres_spacing)*4)),
                                 int(round(np.min(fix_lowres_spacing)*16))]}

    global_affine_kwargs : dict
        Any arguments you would like to pass to the global instance of
        bigstream.align.affine_align. See the docstring for that function
        for valid parameters.
        default : {'alignment_spacing':np.min(fix_lowres_spacing)*4,
                   'shrink_factors':(2,),
                   'smooth_sigmas':(np.min(fix_lowres_spacing)*8,),
                   'optimizer_args':{
                       'learningRate':0.25,
                       'minStep':0.,
                       'numberOfIterations':400,
                   }}

    local_ransac_kwargs : dict
        Any arguments you would like to pass to the local instances of
        bigstream.align.feature_point_ransac_affine_align. See the
        docstring for that function for valid parameters.
        default : {'blob_sizes':same physical size as global_ransac values,
                                but scaled to highres voxel grid}

    local_deform_kwargs : dict
        Any arguments you would like to pass to the local instances of
        bigstream.align.deformable_align. See the docstring for that function
        for valid parameters.
        default : {'smooth_sigmas':(np.min(fix_highres_spacing)*2,),
                   'control_point_spacing':np.min(fix_highres_spacing)*128,
                   'control_point_levels':(1,),
                   'optimizer_args':{
                       'learningRate':0.25,
                       'minStep':0.,
                       'numberOfIterations':25,
                   }}

    cluster_kwargs : dict
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClusterWrap.local_cluster. This is how
        distribution parameters are specified.

    cluster : dask cluster object
        Only set if you have constructed your own static cluster. The default
        behavior is to construct a cluster for the duraction of this pipeline,
        then close it when the function is finished. However, if you provide
        a cluster through this keyword then all distributed computations will
        occur on that cluster

    Returns
    -------
    affine : 4x4 ndarray
        The global affine transform

    deform : zarr array
        The local displacement vector field (assumed too large for memory)

    aligned : zarr array
        mov_highres resampled to match fix_highres, i.e. application of
        [affine, deform] to mov_highres (assumed too large for memory)
               
    """

    # ensure lowres datasets are in memory
    fix_lowres = fix_lowres[...]
    mov_lowres = mov_lowres[...]

    # configure global affine alignment at lowres
    alignment_spacing = np.min(fix_lowres_spacing)*4
    blob_min = int(round(np.min(fix_lowres_spacing)*4))
    blob_max = int(round(np.min(fix_lowres_spacing)*16))
    a = {'alignment_spacing':alignment_spacing,
         'blob_sizes':[blob_min, blob_max]}
    b = {'alignment_spacing':alignment_spacing,
         'shrink_factors':(2,),
         'smooth_sigmas':(2*alignment_spacing,),
         'optimizer_args':{
             'learningRate':0.25,
             'minStep':0.,
             'numberOfIterations':400,
         },
    }

    steps = [
        ('ransac', {**a, **global_ransac_kwargs}),
        ('affine', {**b, **global_affine_kwargs}),
    ]
    if no_global_affine:
        steps = [
            ('ransac', {**a, **global_ransac_kwargs})
        ]
    if overwrite_lowres or (not os.path.exists(f'{write_directory}/{fname}_affine.mat') and
                            not os.path.exists(f'{write_directory}/{fname}_affine.npy')):
    
        # run global affine alignment at lowres
        affine = alignment_pipeline(
            fix_lowres, mov_lowres,
            fix_lowres_spacing, mov_lowres_spacing,
            steps=steps,
        )
        
        # apply global affine and save result
        aligned = apply_transform(
            fix_lowres, mov_lowres,
            fix_lowres_spacing, mov_lowres_spacing,
            transform_list=[affine,],
        )

        
        np.savetxt(f'{write_directory}/{fname}_affine.mat', affine)
        np.save(f'{write_directory}/{fname}_affine.npy', aligned)

        tif_imwrite(f'{write_directory}/{fname}_mov_lowres.tiff', mov_lowres.transpose(2,1,0))
        tif_imwrite(f'{write_directory}/{fname}_fix_lowres.tiff', fix_lowres.transpose(2,1,0))
        tif_imwrite(f'{write_directory}/{fname}_both.tiff', np.swapaxes(np.array([ aligned.transpose(2,1,0), 
                                                                            fix_lowres.transpose(2,1,0)]),0,1),
                                                                            imagej=True)
        tif_imwrite(f'{write_directory}/{fname}_mov_lowres_affine.tiff', aligned.transpose(2,1,0))

        if only_lowres:
            return affine, aligned, None
    else:
        print("low res was already calculated loading files",
                f'{write_directory}/{fname}_affine.mat',
                f'{write_directory}/{fname}_affine.npy')
        affine = np.loadtxt(f'{write_directory}/{fname}_affine.mat')
        aligned = np.load(f'{write_directory}/{fname}_affine.npy')


    # configure local deformable alignment at highres
    ratio = np.min(fix_lowres_spacing) / np.min(fix_highres_spacing)
    blob_min = int(round(blob_min * ratio))
    blob_max = int(round(blob_max * ratio))
    a = {'blob_sizes':[blob_min, blob_max]}
    b = {'smooth_sigmas':(2*np.min(fix_highres_spacing),),
         'control_point_spacing':np.min(fix_highres_spacing)*128,
         'control_point_levels':(1,),
         'optimizer_args':{
             'learningRate':0.25,
             'minStep':0.,
             'numberOfIterations':25,
         },
    }

    steps = [
        ('ransac', {**a, **local_ransac_kwargs}),
        ('deform', {**b, **local_deform_kwargs}),
    ]

    if no_deform == True:
        steps = [
            ('ransac', {**a, **local_ransac_kwargs}),
        ]
    # closure for distributed functions
    alignment = lambda x: distributed_piecewise_alignment_pipeline(
        fix_highres, mov_highres,
        fix_highres_spacing, mov_highres_spacing,
        steps=steps,
        blocksize=blocksize,
        static_transform_list=[affine,],
        write_path=write_directory + '/deform.zarr',
        cluster=x,
    )
    resample = lambda x: distributed_apply_transform(
        fix_highres, mov_highres,
        fix_highres_spacing, mov_highres_spacing,
        transform_list=[affine, deform],
        blocksize=blocksize,
        write_path=write_directory + '/deformed.zarr',
        cluster=x,
    )

    # if no cluster was given, make one then run on it
    if cluster is None:
        print('constructing cluster')
        with cluster_constructor(**{**c, **cluster_kwargs}) as cluster:
            print('running alignment')
            deform = alignment(cluster)
            aligned = resample(cluster)
    # otherwise, use the cluster that was given
    else:
        deform = alignment(cluster)
        aligned = resample(cluster)

    return affine, deform, aligned

def get_registration_score(fixed, mov):
    fixed_image = sitk.GetImageFromArray(fixed.astype(np.float32))
    registered_image = sitk.GetImageFromArray(mov.astype(np.float32))
    # Initialize the registration method
    irm = sitk.ImageRegistrationMethod()

    # Set the metric to ANTS Neighborhood Correlation
    irm.SetMetricAsMattesMutualInformation()  # 4 is the radius of the neighborhood

    # Set the fixed and moving images for the metric evaluation
    out = irm.MetricEvaluate(fixed_image,registered_image)
    return out

# @lru_cache(maxsize=128)
def register_lowres(
    fix_lowres,
    mov_lowres,
    fix_lowres_spacing,
    mov_lowres_spacing,
    write_directory,
    global_ransac_kwargs={},
    fname='',
    write_only_aligned=True,
):
    """
    Function to register lowres images in paramters scan, this function is used by 1_scan_lowres_parameters.ipynb
    """

    # ensure lowres datasets are in memory
    fix_lowres = fix_lowres[...]
    mov_lowres = mov_lowres[...]

    # configure global affine alignment at lowres
    alignment_spacing = np.min(fix_lowres_spacing)*4
    blob_min = int(round(np.min(fix_lowres_spacing)*4))
    blob_max = int(round(np.min(fix_lowres_spacing)*16))
    #print(f'1, {blob_min=} , {blob_max=}')
    a = {'alignment_spacing':alignment_spacing,'blob_sizes':[blob_min, blob_max]}
    
    #numberOfIterations = 10 instead of 100
    global_ransac_kwargs_full = {**a, **global_ransac_kwargs}

    affine = feature_point_ransac_affine_align(fix_lowres, mov_lowres, 
                                                fix_lowres_spacing, mov_lowres_spacing, 
                                                safeguard_exceptions=False,
                                                **global_ransac_kwargs_full)

    
    if write_only_aligned:
        # if affine is not a solution, return None
        if (np.eye(fix_lowres.ndim + 1) == affine).all():
            return None
        
    # apply global affine and save result
    aligned = apply_transform(
        fix_lowres, mov_lowres,
        fix_lowres_spacing, mov_lowres_spacing,
        transform_list=[affine],
    )

    reg_score = get_registration_score(aligned, fix_lowres)
    reg_score_text = str(np.round(reg_score,3)).replace('-','m')
    print(f'{write_directory}/{reg_score_text}_{fname}_both.tiff',flush=True)
    tif_imwrite(f'{write_directory}/{reg_score_text}_{fname}_both.tiff', np.swapaxes(np.array([ aligned.transpose(2,1,0), 
                                                                        fix_lowres.transpose(2,1,0)]),0,1),
                                                                        imagej=True)
    
    return aligned
