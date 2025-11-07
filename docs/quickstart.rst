Quick Start
===========

This guide will help you run your first pipeline.

Basic Usage
-----------

The pipeline is controlled by a manifest file in HJSON format that describes your experiment.

Running the Pipeline
--------------------

1. **Prepare your manifest file**

   Create a copy of the example manifest:

   .. code-block:: bash

      cp examples/demo.json examples/my_experiment.hjson

   Edit the file to match your experiment parameters.

2. **Organize your data**

   Follow the file structure described in ``examples/file_structure.txt``. Copy your HCR and 2-photon data to the appropriate folders.

3. **Run the full pipeline**

   .. code-block:: bash

      python master_pipeline.py --manifest examples/my_experiment.hjson

Pipeline Options
----------------

The pipeline supports several command-line options:

HCR-only Processing
~~~~~~~~~~~~~~~~~~~

To run only the HCR processing steps (skipping two-photon processing):

.. code-block:: bash

   python master_pipeline.py --manifest examples/my_experiment.hjson --only_hcr

Multi-plane Processing
~~~~~~~~~~~~~~~~~~~~~~

To process additional functional planes beyond the reference plane:

.. code-block:: bash

   python master_pipeline.py --manifest examples/my_experiment.hjson --add_planes

Example Workflows
-----------------

Full Pipeline with High-Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For experiments with both high-resolution anatomical tiles and functional imaging:

.. code-block:: bash

   python master_pipeline.py --manifest examples/CIM132.hjson

This will:

1. Process high-resolution SBX tiles
2. Unwarp and stitch tiles
3. Extract Suite2p registered planes
4. Register HCR rounds
5. Run Cellpose segmentation
6. Extract intensities and align masks

HCR-Only Pipeline
~~~~~~~~~~~~~~~~~

For experiments with only HCR data:

.. code-block:: bash

   python master_pipeline.py --manifest examples/my_experiment.hjson --only_hcr

This will:

1. Register HCR rounds
2. Run Cellpose segmentation
3. Extract probe intensities
4. Align and merge masks

Next Steps
----------

- Learn more about the :doc:`pipeline_overview`
- Explore the :doc:`api/index` for detailed function reference
- Check the ``examples/`` directory for sample manifest files
