Pipeline steps:



PreProcessing - Manually:
- create a copy of examples/demo.json and change the variable to match the new experiment, save it as mouse_name.json
- copy the relevant files (HCR/2P) to the relevant folder (look at examples/file_structure.txt)
- run the pipeline with the new json file. Running everything on colab00 - in terminal run python master_pipeline.py examples/CIM132.hjson

A - Automatic
M - Manual
SA - Semi-Automatic

1. parse manifest
2. validate manifest
3. process sbx files (A):
    - transform sbx to tiff
    - output files:
        - currently outputs all planes (Warped)
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
