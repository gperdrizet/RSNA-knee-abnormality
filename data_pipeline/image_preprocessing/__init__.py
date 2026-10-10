'''
Image preprocessing stage: align (RAS), normalize (percentile clip ->
z-score -> uint8), and resample (1 mm isotropic) the 3D knee MRI
series, producing NIfTI volumes + a manifest joined with the study-level
pseudo labels for downstream pseudo-labeling / model training.

See config.py for the constants and their rationale, and
docs/IMAGE_PREPROCESSING_PLAN.md for the full design.
'''
