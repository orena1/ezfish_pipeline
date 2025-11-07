Installation
============

Requirements
------------

The EZfish Pipeline requires Python 3.8 or later.

Installing Dependencies
-----------------------

1. Clone the repository:

.. code-block:: bash

   git clone <repository-url>
   cd ezfish_pipeline

2. Install the required Python packages:

.. code-block:: bash

   pip install -r requirements

3. Install documentation dependencies (optional, for building docs):

.. code-block:: bash

   pip install -r docs/requirements.txt

Required Packages
-----------------

The pipeline depends on the following packages:

- **suite2p**: Two-photon imaging analysis
- **cellpose**: Cell segmentation
- **tifffile**: TIFF file handling
- **rich**: Terminal formatting and progress bars
- **tqdm**: Progress bars

Verification
------------

To verify the installation, run:

.. code-block:: bash

   python master_pipeline.py --help

You should see the help message with available command-line options.
