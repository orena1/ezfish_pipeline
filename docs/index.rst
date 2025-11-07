EZfish Pipeline Documentation
=============================

Welcome to the EZfish Pipeline documentation. This pipeline processes two-photon imaging and HCR (Hybridization Chain Reaction) data for imaging neuroscience experiments.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   pipeline_overview
   api/index

Overview
--------

The EZfish Pipeline is designed to process multi-modal imaging data, including:

- Two-photon functional imaging
- High-resolution anatomical tiles
- HCR fluorescence rounds
- Cell segmentation using Cellpose
- Image registration and alignment

Key Features
------------

- **Automated Processing**: End-to-end pipeline from raw data to analyzed results
- **Multi-plane Support**: Process multiple functional planes with reference-based alignment
- **Flexible Configuration**: HJSON-based manifest files for easy experiment setup
- **Modular Design**: Individual pipeline steps can be run independently

Quick Links
-----------

- :doc:`installation` - Get started with installation
- :doc:`quickstart` - Run your first pipeline
- :doc:`pipeline_overview` - Understand the pipeline steps
- :doc:`api/index` - API reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
