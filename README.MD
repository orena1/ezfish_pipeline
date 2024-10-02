Pipeline steps:



PreProcessing - Manually:
- create a copy of examples/demo.json and change the variable to match the new experiment, save it as mouse_name.json
- copy the relevant files (HCR/2P) to the relevant folder (look at examples/file_structure.txt)
- run the pipeline with the new json file. Running everything on colab00 - in terminal run python -- manifest master_pipeline.py examples/CIM132.hjson

A - Automatic
M - Manual
SA - Semi-Automatic


# Steps of the pipeline, will be instructed to the manual stesps 
1. parse manifest. I.e: python master_pipeline.py  --manifest examples/CIM130.hjson
2. validate manifest
2. generate mean images from low res functional video 
    -extract meanE, meanOverTime, and meanImg from suite2p files. 
3. process high res tile sbx files (A):
    - transform sbx to tiff
    - output files:
        - currently outputs all planes (Warped),
4. unwarp tiff files (A):
    - take each plane and use remap & unwarp_config to fix the x axis stretch from the resonant
    - output files:
        - currently outputs all planes (Unwarped)
5. stitch tiff files (M):
    - take each plane and stitch them together manually using bigsticher
6. register HCR rounds - ezfish_pipeline scripts (SA)
    - run 1_scan_lowres_parameters.ipynb with the correct hjson file
    - find the correct paramters that are good for lowres
    - run_2_scan_highres_parameters.ipynb with the correct hjson file
    - find the correct paramters that are good for highres
    - set the correct parameters in the OUTPUT/params.hjson file
    - apply registration to all channels in rounds
7. 
