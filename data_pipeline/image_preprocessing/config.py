'''
Configuration constants for the image preprocessing stage.

Conventions
-----------
- Output volumes are 3D NIfTI (``.nii.gz``), reoriented to ``RAS``
  (x=right, y=anterior, z=superior - the NIfTI standard) and resampled to
  a 1.0 mm isotropic grid on a FIXED, per-plane target box (see
  ``TARGET_BOX_VOXELS``): every series of a given anatomical plane has
  the exact same output shape, so series never mix planes and downstream
  batching sees one tensor shape per plane.  Boxes were sized from a
  2,400-series extent scan (800/plane, exact DICOM slice positions):
  p99.9 per world axis + 4 mm margin, rounded to even voxels.
  Series whose world extent exceeds the box (minus the pad ring) are
  flagged ``failed`` in the manifest rather than cropped - a handful of
  extended-FOV acquisitions are expected to land there.
  1 mm was chosen from a 120-series metadata scan: native in-plane
  spacing spans 0.25-0.625 mm and through-plane 2.0-4.6 mm, so 1 mm
  keeps full in-plane detail of the majority of series without blowing
  up volume size.
- Intensities are normalized per volume: clipped to the 0.5/99.5
  percentiles, z-scored on non-zero (foreground) voxels, clipped to
  +/-3 sigma, and quantized to uint8 [0, 255].  The inversion
  (z = u8/255 * 6 - 3; x = z * std + mean) is recoverable from the
  per-series sidecar JSON and the manifest.
- Storage: uint8 keeps the full train set (~24.4k series) to ~25-35 GB
  gzipped instead of ~120+ GB as float32.
'''

import os
from pathlib import Path

# Repository layout.  Inputs default to the repo's data/ dir (symlinked or
# mounted to the raw DICOM tree, as in the dev container) but can be pointed
# anywhere via env vars, e.g. in a host shell without the symlinks:
#   RSNA_RAW_DATA_DIR=/mnt/fast_scratch/RSNA-knee-data/raw
REPO_ROOT           = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR        = Path(os.environ.get(
    'RSNA_RAW_DATA_DIR', str(REPO_ROOT / 'data' / 'raw')))
TRAIN_SERIES_CSV    = RAW_DATA_DIR / 'train_series.csv'
TEST_SERIES_CSV     = RAW_DATA_DIR / 'test_series.csv'
TRAIN_SERIES_DIR    = RAW_DATA_DIR / 'train_series'
TEST_SERIES_DIR     = RAW_DATA_DIR / 'test_series'

# Study-level pseudo labels (output of the label extraction stage) - joined
# into the manifest by StudyInstanceUID.
PSEUDOLABELED_CSV   = Path(os.environ.get(
    'RSNA_PSEUDOLABELED_CSV',
    str(REPO_ROOT / 'data' / 'pseudolabeled-train.csv')))

# Output root: data/pipeline/image_preprocessing/<run>/ by convention.
# The full train+test output is ~25-35 GB; if the data mount does not have
# the space, point this at a larger mount, e.g.:
#   IMAGE_PREPROCESSING_OUT=/mnt/fast_scratch/rsna-preprocessed
OUTPUT_ROOT = Path(os.environ.get(
    'IMAGE_PREPROCESSING_OUT',
    str(REPO_ROOT / 'data' / 'pipeline' / 'image_preprocessing')
))

# Splits processed by the pipeline
SPLITS = ('train', 'test')

# --- Geometry ------------------------------------------------------------
# Standard anatomical orientation: RAS (x=right, y=anterior, z=superior).
TARGET_ORIENTATION = 'RAS'

# Resampling target: 1.0 mm isotropic.
VOXEL_SPACING_MMM = (1.0, 1.0, 1.0)

# Fixed per-plane output boxes in target voxels (x, y, z) RAS order,
# INCLUDING the padding ring.  All series of a plane share the box, so
# the output shape is standardized per plane with a set number of
# slices.  Anatomy is centered in the box (see preprocess.py).
#
# Sizing: 2,400-series extent scan (800/plane, exact slice positions,
# seed=0) - p99.9 per world axis + 4 mm margin, rounded up to even.
# Expected flag rate (extent beyond box minus pad ring) from the scan:
# Sagittal ~0.5%, Coronal ~1.4%, Axial ~1.5% - ~250 series (~45 studies
# lose one sequence, the rest of their sequences are unaffected).
TARGET_BOX_VOXELS = {
    'Sagittal': (218, 286, 278),
    'Coronal':  (268, 208, 270),
    'Axial':    (308, 358, 244),
}

# Empty padding ring inside the target box, in target voxels
# (8 voxels = 8 mm), so no anatomy touches the volume border.
GRID_PAD_VOXELS = 8

# Resample interpolation for intensity volumes.
# (sitk.sitkLinear - imported in preprocess.py to keep this module
# import-light.)

# Reconstruction sanity check: the through-plane extent implied by the DICOM
# slice positions must agree with (n_slices - 1) * slice_thickness within
# this relative tolerance, otherwise the series is flagged failed rather
# than resampled onto a corrupted grid.
Z_EXTENT_TOLERANCE = 0.25

# --- Normalization ---------------------------------------------------------
# Percentile clip bounds (applied to the resampled volume, background
# included - p0.5 is 0 for essentially all series).
CLIP_PERCENTILE_LOW  = 0.5
CLIP_PERCENTILE_HIGH = 99.5

# Z-scoring statistics are taken on non-zero voxels (foreground), matching
# nnU-Net convention, so background lands at a consistent negative value
# across scans regardless of how much air is in the frame.

# Robust clip on the z-score before 8-bit quantization.
ZCLIP_SIGMA = 3.0

# Output pixel type. uint8 is lossless up to the 1/255 quantization step
# and 4x smaller than float32 on disk.
OUTPUT_DTYPE = 'uint8'

# --- Runtime ---------------------------------------------------------------
# Per-task SimpleITK thread cap (Luigi workers x this should stay <= ncpu).
SIMPLEITK_THREADS = 3

# Output file names
NIFTI_SUFFIX      = '.nii.gz'
SIDECAR_SUFFIX    = '.json'
MANIFEST_NAME     = 'manifest.csv'
RUN_META_NAME     = 'run_meta.json'
FAILED_LOG_NAME   = 'failed_series.txt'
