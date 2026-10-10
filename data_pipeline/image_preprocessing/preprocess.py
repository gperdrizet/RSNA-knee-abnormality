'''
Core preprocessing transforms: orientation, resampling, normalization.

Pipeline per series (see config for the constants and why):

    DICOM dir  ->  volume.read_series_volume   (native LPS grid, rescaled)
               ->  reorient                     (RAS)
               ->  fixed-box check              (world extent must fit the
                                                 plane's TARGET_BOX_VOXELS,
                                                 else the series is failed)
               ->  resample_to_standard         (1 mm iso, onto the fixed
                                                 per-plane box, centered)
               ->  normalize_volume             (percentile clip -> z-score
                                                 on foreground -> +/-3 sigma
                                                 -> uint8)

The functions here are pure (image in, image/params out) so the Luigi
tasks in tasks.py and the analysis notebooks can share them.
'''

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from data_pipeline.image_preprocessing import config, volume as _volume

# Cap per-task ITK threads (Luigi workers x this should stay <= ncpu).
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(config.SIMPLEITK_THREADS)


def reorient(image: sitk.Image,
             orientation: str = config.TARGET_ORIENTATION) -> sitk.Image:
    '''Reorient an image so its axes follow ``orientation`` (RAS:
    x=right, y=anterior, z=superior).  Pure axis permutation/flip -
    no interpolation.'''

    return sitk.DICOMOrient(image, orientation)


def world_extents_mmm(image: sitk.Image) -> np.ndarray:
    '''World-space extent of the volume's voxel footprint, in mm, per
    world axis (x, y, z).

    The image may be oblique (DICOMOrient only permutes/flips axes), so
    the extent along world axis a is the sum of the absolute projections
    of every volume axis onto it.  Using per-axis size*spacing directly
    (axis-aligned assumption) underestimates the extent by 1-4 mm on
    oblique series.
    '''

    size = np.array(image.GetSize(), dtype=float)
    src_spacing = np.array(image.GetSpacing(), dtype=float)
    direction = np.array(image.GetDirection(), dtype=float).reshape(3, 3)

    return (np.abs(direction) * (size * src_spacing)[None, :]).sum(axis=1)


def resample_to_standard(image: sitk.Image,
                         spacing: tuple[float, float, float]
                         = config.VOXEL_SPACING_MMM,
                         pad_voxels: int = config.GRID_PAD_VOXELS,
                         target_size: tuple[int, int, int] | None = None
                         ) -> sitk.Image:
    '''Resample onto a centered, strictly axis-aligned RAS grid.

    The output grid has an identity (RAS) direction and shares the
    volume's center, so the anatomy sits in the middle of the volume on
    a grid whose axes are exactly x=right, y=anterior, z=superior.  Any
    residual in-plane obliquity from the acquisition is removed by the
    resample; the grid extent accounts for that obliquity (sum of
    absolute axis projections).  Linear interpolation.

    Grid size: with ``target_size`` given (the normal pipeline path) the
    output is the fixed per-plane box, so every series of a plane has
    the same shape; the caller must first verify the volume's world
    extent fits (preprocess_series does).  Without ``target_size`` the
    grid spans the volume's own extent plus ``pad_voxels`` of padding
    (kept for ad-hoc notebook use).

    The image must already be reoriented (reorient()) - otherwise its
    axes may point the wrong way (e.g. left instead of right).
    '''

    size = np.array(image.GetSize(), dtype=float)

    # True center: the midpoint of the voxel-center lattice is (size-1)/2,
    # not size/2 (which is half a voxel off per axis - up to ~half the
    # through-plane spacing in z).
    center = np.array(
        image.TransformContinuousIndexToPhysicalPoint(
            ((size[0] - 1) / 2.0, (size[1] - 1) / 2.0, (size[2] - 1) / 2.0)
        )
    )

    if target_size is None:
        extents_mmm = world_extents_mmm(image)
        new_size = [
            int(np.ceil(ext / sp)) + 2 * pad_voxels
            for ext, sp in zip(extents_mmm, spacing)
        ]
    else:
        new_size = [int(n) for n in target_size]
        if any(n <= 2 * pad_voxels for n in new_size):
            raise ValueError(
                f'target box {new_size} too small for pad ring '
                f'({2 * pad_voxels} voxels)'
            )

    # The output grid is strictly RAS (identity direction).  Its origin
    # is the volume center shifted by half the grid extents along the
    # grid axes - a plain per-coordinate subtraction, which is valid
    # precisely because the grid axes are the world axes.
    half_extents = np.array([(n - 1) * sp / 2 for n, sp in
                             zip(new_size, spacing)])
    out_origin = center - half_extents

    ref = sitk.Image(new_size[0], new_size[1], new_size[2],
                     sitk.sitkFloat32)
    ref.SetSpacing(spacing)
    ref.SetOrigin(out_origin)
    ref.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    # No nearest-neighbor extrapolation: sample points outside the source
    # volume (the padding ring) must be the empty background (0), not the
    # replicated edge voxel - nearest-neighbor extrapolation painted a
    # fake high-intensity halo around the anatomy in the pad ring.
    return sitk.Resample(
        image, ref,
        interpolator=sitk.sitkLinear,
        defaultPixelValue=0.0,
        outputPixelType=sitk.sitkFloat32,
    )


def normalize_volume(arr: np.ndarray,
                     clip_low_pct: float = config.CLIP_PERCENTILE_LOW,
                     clip_high_pct: float = config.CLIP_PERCENTILE_HIGH,
                     zclip: float = config.ZCLIP_SIGMA
                     ) -> tuple[np.ndarray, dict]:
    '''Normalize a float volume to uint8 [0, 255].

    Steps:
      1. clip to the [clip_low_pct, clip_high_pct] percentiles,
      2. z-score using mean/std of non-zero (foreground) voxels,
      3. clip to [-zclip, +zclip] sigma,
      4. linear map to 0..255.

    Returns (uint8_array, params).  params carries everything needed to
    invert the transform (x = ((u8/255)*2*zclip - zclip) * std + mean)
    and to spot-check the statistics: clip_low/clip_high (the actual
    percentile values), mean, std, zclip.
    '''

    low, high = np.percentile(arr, [clip_low_pct, clip_high_pct])
    clipped = np.clip(arr, low, high)

    fg = clipped > 0
    n_fg = int(fg.sum())

    if n_fg == 0:
        raise ValueError('volume is empty (all-zero)')

    mean = float(clipped[fg].mean())
    std = float(clipped[fg].std())

    if not np.isfinite(std) or std < 1e-8:
        raise ValueError(f'zero intensity variance (mean={mean:.1f})')

    z = (clipped - mean) / std
    z = np.clip(z, -zclip, zclip)

    u8 = np.round(z / (2 * zclip) * 255.0 + 127.5).astype(np.uint8)

    params = {
        'clip_low': round(float(low), 4),
        'clip_high': round(float(high), 4),
        'fg_mean': round(mean, 4),
        'fg_std': round(std, 4),
        'zclip_sigma': zclip,
        'n_fg_voxels': n_fg,
    }

    return u8, params


def _wrap_as_image(arr_u8: np.ndarray, geometry: sitk.Image) -> sitk.Image:
    '''Wrap a (z, y, x) uint8 array as a SimpleITK image sharing
    ``geometry``'s origin/spacing/direction.'''

    img = sitk.GetImageFromArray(arr_u8)
    img.SetOrigin(geometry.GetOrigin())
    img.SetSpacing(geometry.GetSpacing())
    img.SetDirection(geometry.GetDirection())

    return img


def preprocess_series(series_path: str, plane: str) -> tuple[sitk.Image, dict]:
    '''One-call preprocessing for one series directory.

    Returns (uint8_image, record): the image is RAS, 1 mm isotropic,
    uint8, on the fixed per-plane box from config.TARGET_BOX_VOXELS;
    record holds geometry + normalization + source provenance for the
    per-series sidecar / manifest.  Raises ValueError for an unknown
    plane and SeriesReconstructionError when the volume's world extent
    exceeds the fixed box (those series are flagged, never cropped).
    '''

    box = config.TARGET_BOX_VOXELS.get(str(plane).strip())

    if box is None:
        raise ValueError(
            f'unknown anatomical plane {plane!r}; expected one of '
            f'{sorted(config.TARGET_BOX_VOXELS)}'
        )

    vol, src_meta = _volume.read_series_volume(series_path)

    oriented = reorient(vol)

    # Fixed-box check: the anatomy must fit inside the box minus the pad
    # ring, otherwise resampling would silently clip it.
    extents = world_extents_mmm(oriented)
    budget = (
        np.array([b - 2 * config.GRID_PAD_VOXELS for b in box])
        * np.array(config.VOXEL_SPACING_MMM)
    )

    over = [
        f'{ax} {ext:.1f}mm > {bud:.0f}mm'
        for ax, ext, bud in zip('xyz', extents, budget) if ext > bud
    ]

    if over:
        raise _volume.SeriesReconstructionError(
            f'world extent exceeds fixed {plane} box {box}: '
            + ', '.join(over)
        )

    resampled = resample_to_standard(oriented, target_size=box)

    arr = sitk.GetArrayFromImage(resampled)
    u8, params = normalize_volume(arr)

    img = _wrap_as_image(u8, resampled)
    out_size = img.GetSize()
    out_spacing = img.GetSpacing()

    record = {
        'status': 'ok',
        'orientation': config.TARGET_ORIENTATION,
        'target_box_voxels': list(box),
        'shape_xyz': list(out_size),
        'spacing_mmm': [round(s, 4) for s in out_spacing],
        'volume': {
            **src_meta,
            'resampled_size_xyz': list(out_size),
            'resampled_extent_mmm': [
                round((s - 1) * sp, 2)
                for s, sp in zip(out_size, out_spacing)
            ],
        },
        'normalization': {
            **params,
            'method': (f'percentile [{config.CLIP_PERCENTILE_LOW}, '
                       f'{config.CLIP_PERCENTILE_HIGH}] -> z-score(fg) -> '
                       f'clip +/-{config.ZCLIP_SIGMA} -> uint8'),
        },
    }

    return img, record


def write_nifti(image: sitk.Image, path: str) -> None:
    '''Write a NIfTI volume (.nii.gz).  The caller handles atomicity.'''

    if not path.endswith('.nii.gz'):
        path = path + '.nii.gz'

    sitk.WriteImage(image, path, useCompression=True, compressionLevel=6)
