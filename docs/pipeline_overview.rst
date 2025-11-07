Pipeline Overview
=================

The EZfish Pipeline consists of several processing stages that transform raw imaging data into analyzed results.

Pipeline Architecture
---------------------

The pipeline is organized into modular components, each handling a specific aspect of data processing:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Description
   * - ``meta``
     - Manifest parsing and validation
   * - ``tiling``
     - High-resolution tile processing and stitching
   * - ``functional``
     - Two-photon functional data extraction
   * - ``registrations``
     - HCR round-to-round image registration
   * - ``segmentation``
     - Cell segmentation and intensity extraction

Processing Stages
-----------------

1. Preprocessing (Manual)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Before running the automated pipeline:

- Create and configure a manifest file from ``examples/demo.json``
- Organize your data files (HCR/2P) following the expected structure
- Ensure all required files are in the correct locations

2. Manifest Parsing
~~~~~~~~~~~~~~~~~~~~

The pipeline begins by parsing and validating the manifest file:

.. code-block:: python

   full_manifest = mt.main_pipeline_manifest(args.manifest)
   specs, has_hires = mt.verify_manifest(full_manifest, args)

3. High-Resolution Processing (Automatic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For experiments with high-resolution anatomical tiles:

**Process SBX Files**

Convert SBX format to TIFF and extract all planes:

.. code-block:: python

   tl.process_session_sbx(full_manifest, session)

**Unwarp Tiles**

Correct X-axis stretch from resonant scanning using remap and unwarp configuration:

.. code-block:: python

   tl.unwarp_tiles(full_manifest, session)

**Stitch and Rotate**

Combine tiles into a single registered volume:

.. code-block:: python

   tl.stitch_tiles_and_rotate(full_manifest, session)

4. Functional Data Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract registered functional planes from Suite2p output:

.. code-block:: python

   fc.extract_suite2p_registered_planes(full_manifest, session)

For experiments without high-resolution data, the red channel can be combined:

.. code-block:: python

   fc.extract_suite2p_registered_planes(full_manifest, session, combine_with_red=True)

5. HCR Registration (Semi-Automatic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Register HCR fluorescence rounds to each other:

**Interactive Parameter Selection**

1. Run ``1_scan_lowres_parameters.ipynb`` to find optimal low-resolution parameters
2. Run ``2_scan_highres_parameters.ipynb`` to find optimal high-resolution parameters
3. Set the final parameters in ``OUTPUT/params.hjson``

**Apply Registration**

.. code-block:: python

   rf.register_rounds(full_manifest)

6. Cell Segmentation
~~~~~~~~~~~~~~~~~~~~~

**Run Cellpose**

Segment cells in HCR images using Cellpose:

.. code-block:: python

   sg.run_cellpose(full_manifest)

**Extract Intensities**

Extract probe intensities from segmented cells:

.. code-block:: python

   sg.extract_probs_intensities(full_manifest)

7. Integration with Functional Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Extract 2P Cellpose Masks**

Apply Cellpose segmentation to two-photon images:

.. code-block:: python

   sg.extract_2p_cellpose_masks(full_manifest, session)

**Extract Electrophysiology Intensities**

Extract functional signals from segmented ROIs:

.. code-block:: python

   sg.extract_electrophysiology_intensities(full_manifest, session)

8. Mask Alignment and Merging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Align Masks**

Align masks between HCR and functional imaging:

.. code-block:: python

   sg.align_masks(full_manifest, session, only_hcr=args.only_hcr, reference_plane=reference_plane)

**Merge Masks**

Combine aligned masks into single output files:

.. code-block:: python

   sg.merge_masks(full_manifest, session, only_hcr=args.only_hcr)

Multi-Plane Processing
-----------------------

The pipeline supports processing multiple functional planes:

**Reference-Based Mode** (``--add_planes``)

Process additional planes relative to a reference plane:

.. code-block:: python

   reference_plane = session['functional_plane'][0]
   functional_planes = session['additional_functional_planes']

**Independent Mode**

Process each plane independently:

.. code-block:: python

   functional_planes = list(session['functional_plane'])

Data Flow
---------

.. code-block:: text

   Raw Data
      ├── Two-Photon SBX Files ──┐
      │   ├── High-res tiles     │
      │   └── Functional data    │
      │                          │
      └── HCR Rounds ────────────┤
                                 │
   Preprocessing               │
      ├── Tile stitching        │
      ├── Unwarping             │
      └── Suite2p extraction    │
                                 │
   Registration                │
      ├── HCR round-to-round    │
      └── Parameter optimization│
                                 │
   Segmentation                │
      ├── Cellpose              │
      └── ROI extraction        │
                                 │
   Analysis                    │
      ├── Intensity extraction  │
      ├── Mask alignment        │
      └── Mask merging          │
                                 ↓
   Final Output
      ├── Aligned masks
      ├── Intensity measurements
      └── Registered volumes

Output Files
------------

The pipeline generates various output files organized by processing stage:

- **Stitched volumes**: High-resolution anatomical images
- **Registered HCR rounds**: Aligned fluorescence data
- **Cellpose masks**: Cell segmentation results
- **Intensity tables**: Extracted probe and functional signals
- **Aligned masks**: Integrated HCR and functional ROIs

See Also
--------

- :doc:`api/index` - Detailed API documentation for each module
- :doc:`quickstart` - Quick start guide with examples
