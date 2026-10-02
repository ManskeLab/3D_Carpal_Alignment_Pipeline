# 3D_CARPAL_ALIGNMENT_PIPELINE/python-files

The Jupyter notebook in this folder goes through the mesh pre-processing and quantification of bone properties required to obtain carpal alignment measurement outputs.

Once the Conda environment has been installed (from 3DCAP.yaml), the carpal alignment pipeline can be run. Filepaths for the radius, ulna, scaphoid, lunate, capitate, and third metacarpal must be input for execution of the pipeline. There are five values that may require manual input prior to complete execution of the pipeline:

- **SLIL_INJURY (Cell 4 - Global Variables)**: True or False - This should be set to True when evaluating participants with an SLIL injury
- **RADIUS_SHAFT_DIRECTION (Cell 4 - Global Variables)**: "X", "Y", or "Z" - This should be adjusted to be aligned with the coordinates of the global coordinate system. For example, this should be set to "Y" for datasets where the radius shaft is parallel to the global Y-axis.
- **HAND (Cell 5 - Load the carpal bone meshes)**: "L" or "R" - This variable must be adjusted depending on the side being evaluated.
- **TRANSFORM_RADIUS (Cell 5 - Load the carpal bone meshes)**: True or False - This should be set to True if there is not sufficient length of the radius in a scan. In this case, there must be a another mesh of the radius that can be transformed to the position of the short-length radius.
- **SEGMENTATION_FROM_ITK_SNAP (Cell 5 - Load the carpal bone meshes)**: True or False - If segmentation was obtained using the ITK-SNAP software (e.g. the nnInteractive module), this should be set to True

