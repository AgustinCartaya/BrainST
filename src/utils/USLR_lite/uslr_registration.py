"""Active registration backend used by the preprocessing pipeline.

Implements a lightweight ("lite") variant of USLR-style rigid/affine
registration to the MNI152 template: anatomical landmark centroids are
computed from a SynthSeg segmentation, then a closed-form affine that
aligns those centroids to the atlas's centroids is solved for and applied
via SimpleITK resampling.
"""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk

uslr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))

MNI_IMG_PATH = os.path.join(uslr_path, "data", "atlas", "mni_icbm152_t1norm_tal_nlin_sym_09a.nii.gz")
MNI_SEG_PATH = os.path.join(uslr_path, "data", "atlas", "mni_icbm152_synthseg_tal_nlin_sym_09a.nii.gz")

labels_registration = os.path.join(uslr_path, "data", "labels_classes_priors", "label_list_registration.npy")
"""Path to the ``.npy`` file listing which segmentation label values are
used as registration landmarks (a curated subset chosen for stability
across subjects, e.g. subcortical structures)."""


def compute_centroids_ras(seg_file: str, labelfile: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute the world-space (RAS) centroid of each registration landmark label.

    Args:
        seg_file: Path to a segmentation NIfTI file.
        labelfile: Path to a ``.npy`` array of label values to compute
            centroids for (see :data:`labels_registration`).

    Returns:
        A tuple ``(centroids, valid_mask)`` where:
            - ``centroids`` has shape ``(3, num_labels)`` and holds the
              RAS-space ``(x, y, z)`` centroid of each label.
            - ``valid_mask`` has shape ``(num_labels,)`` and is 1 for
              labels with more than 50 voxels in ``seg_file`` (considered
              reliable enough to use), 0 otherwise.
    """
    segmentation_proxy = nib.load(seg_file)
    segmentation = np.array(segmentation_proxy.dataobj)
    labels = np.load(labelfile)

    num_labels = len(labels)
    # Homogeneous voxel-space centroids (row 3 is the homogeneous "1").
    centroids_voxel = np.zeros([4, num_labels])

    valid_mask = np.ones(num_labels)
    for label_index in range(num_labels):
        voxel_coords = np.where(segmentation == labels[label_index])
        if len(voxel_coords[0]) > 50:
            # Median (rather than mean) centroid: robust to irregularly
            # shaped or partially mis-segmented structures.
            centroids_voxel[0, label_index] = np.median(voxel_coords[0])
            centroids_voxel[1, label_index] = np.median(voxel_coords[1])
            centroids_voxel[2, label_index] = np.median(voxel_coords[2])
            centroids_voxel[3, label_index] = 1
        else:
            valid_mask[label_index] = 0

    # Map voxel-space centroids to world-space (RAS) via the image affine.
    centroids_ras = np.matmul(segmentation_proxy.affine, centroids_voxel)[:-1, :]

    return centroids_ras, valid_mask


def getM(ref: np.ndarray, mov: np.ndarray, use_L1: bool = False) -> np.ndarray:
    """Solve for the affine matrix that best maps points from ``ref`` to ``mov``.

    Fits a 3x4 affine transform (9 linear + 3 translation parameters) in
    the least-squares sense by minimizing ``||M @ ref_i - mov_i||^2``
    summed over all corresponding landmark pairs. This is equivalent to
    an (unconstrained, non-orthogonal) Procrustes-style point-set
    alignment. Optionally, an L1-loss refinement (via linear programming)
    can be used afterward to reduce sensitivity to outlier landmark pairs.

    Args:
        ref: Reference (e.g. atlas/fixed) landmark coordinates, shape
            ``(3, num_landmarks)``.
        mov: Moving (e.g. subject) landmark coordinates, shape
            ``(3, num_landmarks)``, in the same landmark order as ``ref``.
        use_L1: If True, refine the least-squares solution by minimizing
            the L1 (sum of absolute residuals) norm instead, using the
            least-squares solution as the initial guess. More robust to
            a small number of mis-registered landmarks, at higher
            computational cost.

    Returns:
        A 4x4 affine matrix ``M`` (with the last row ``[0, 0, 0, 1]``)
        mapping points from ``ref`` space to ``mov`` space.
    """
    zero_matrix = np.zeros(ref.shape[::-1])
    zero_column = np.zeros([ref.shape[1], 1])
    ones_column = np.ones([ref.shape[1], 1])
    zero_block = np.zeros(zero_matrix.shape)

    # Design matrix encoding, for each of the 3 output components (x, y, z
    # of `mov`), a linear combination of the 9 rotation/scale parameters
    # plus 1 translation parameter.
    design_matrix = np.concatenate(
        [
            np.concatenate([np.transpose(ref), zero_block, zero_block, ones_column, zero_column, zero_column], axis=1),
            np.concatenate([zero_block, np.transpose(ref), zero_block, zero_column, ones_column, zero_column], axis=1),
            np.concatenate([zero_block, zero_block, np.transpose(ref), zero_column, zero_column, ones_column], axis=1),
        ],
        axis=0,
    )

    target_vector = np.concatenate([np.transpose(mov[0, :]), np.transpose(mov[1, :]), np.transpose(mov[2, :])], axis=0)

    # Ordinary least squares via the normal equations.
    solution = np.matmul(
        np.linalg.inv(np.matmul(np.transpose(design_matrix), design_matrix)),
        np.matmul(np.transpose(design_matrix), target_vector),
    )

    if use_L1:
        # Refine using an L1-minimizing linear program, initialized at the
        # least-squares solution, for robustness to outlier correspondences.
        from scipy.optimize import linprog

        positive_slack_constraints = np.concatenate([design_matrix, -np.eye(design_matrix.shape[0])], axis=1)
        negative_slack_constraints = np.concatenate([-design_matrix, -np.eye(design_matrix.shape[0])], axis=1)
        inequality_matrix = np.concatenate([positive_slack_constraints, negative_slack_constraints], axis=0)
        inequality_bound = np.concatenate([target_vector, -target_vector])
        objective_coefficients = np.concatenate([np.zeros(12), np.ones(design_matrix.shape[0])])

        result = linprog(
            objective_coefficients,
            A_ub=inequality_matrix,
            b_ub=inequality_bound,
            method="interior-point",
            bounds=[None, None],
            options={"disp": True, "autoscale": True},
            x0=np.concatenate([solution, 0.1 + np.abs(np.matmul(design_matrix, solution) - target_vector)]),
        )
        solution = result.x[0:12]

    affine_matrix = np.stack(
        [
            [solution[0], solution[1], solution[2], solution[9]],
            [solution[3], solution[4], solution[5], solution[10]],
            [solution[6], solution[7], solution[8], solution[11]],
            [0, 0, 0, 1],
        ]
    )

    return affine_matrix


def ras_to_lps_affine(matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 affine matrix between RAS and LPS coordinate conventions.

    Args:
        matrix: 4x4 affine matrix in one convention.

    Returns:
        The equivalent 4x4 affine matrix in the other convention (flips
        the sign of the X and Y axes; self-inverse operation).
    """
    flip = np.diag([-1, -1, 1, 1])
    return flip @ matrix @ flip


def apply_precomputed_rigid_registration_affine_matrix(
    path_name_img_fixed: str,
    path_name_img_moving: str,
    path_name_img_registered: str,
    affine_matrix: np.ndarray,
    is_label: bool = False,
    verify: bool = False,
    verbose: bool = False,
) -> None:
    """Resample a moving image into a fixed image's grid using a 4x4 affine matrix.

    Args:
        path_name_img_fixed: Path to the fixed/reference image (defines
            the output sampling grid).
        path_name_img_moving: Path to the moving image to resample.
        path_name_img_registered: Path to write the resampled image to.
        affine_matrix: 4x4 affine matrix, in physical (world) coordinates
            using the LPS convention expected by SimpleITK, mapping points
            from the fixed image's space to the moving image's space.
        is_label: If True, use nearest-neighbor interpolation (required
            for integer segmentation/label maps); otherwise linear
            interpolation is used (appropriate for intensity images).
        verify: If True and the output already exists, skip processing.
        verbose: If True, log a message when skipping due to ``verify``.

    Raises:
        ValueError: If ``affine_matrix`` is not a 4x4 numpy array.
    """
    if not isinstance(affine_matrix, np.ndarray) or affine_matrix.shape != (4, 4):
        raise ValueError("Affine matrix must be a numpy array of shape (4, 4).")

    if verify and os.path.exists(path_name_img_registered):
        if verbose:
            print(f"Output path {path_name_img_registered} already exists. Skipping registration.")
        return True

    fixed_image = sitk.ReadImage(path_name_img_fixed, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(path_name_img_moving)

    transform = sitk.AffineTransform(3)
    transform.SetMatrix(affine_matrix[:3, :3].flatten().tolist())
    transform.SetTranslation(affine_matrix[:3, 3].tolist())

    interpolation_mode = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    resampled_image = sitk.Resample(
        moving_image,
        fixed_image,
        transform,
        interpolation_mode,
        0,
        moving_image.GetPixelID(),
    )

    sitk.WriteImage(resampled_image, path_name_img_registered)


def uslr_registration(
    fixed_img_path_name: str,
    fixed_seg_path_name: str,
    moving_img_path_name: str,
    moving_seg_path_name: str,
    out_img_path_name: str,
    out_seg_path_name: str | None = None,
    out_affine_matrix_path_name: str | None = None,
    verify: bool = False,
    verbose: bool = False,
) -> None:
    """Register a moving image (and optionally its segmentation) to a fixed image's space.

    Computes a landmark-centroid-based affine transform from the moving
    segmentation to the fixed segmentation, then resamples the moving
    image (and optionally its segmentation) into the fixed image's grid.

    Args:
        fixed_img_path_name: Path to the fixed (target) image.
        fixed_seg_path_name: Path to the segmentation of the fixed image
            (used to compute reference landmark centroids).
        moving_img_path_name: Path to the moving (source) image.
        moving_seg_path_name: Path to the segmentation of the moving
            image (used to compute landmark centroids to align).
        out_img_path_name: Path to write the registered image to.
        out_seg_path_name: Optional path to also write the moving
            segmentation resampled into the fixed image's space.
        out_affine_matrix_path_name: Optional path to save the computed
            affine transform as a ``.npy`` file.
        verify: If True, skip processing when outputs already exist.
        verbose: If True, log progress/skip messages.

    Side Effects:
        Writes ``out_img_path_name`` and, if requested,
        ``out_seg_path_name`` and ``out_affine_matrix_path_name``.
    """
    centroid_fixed, _ = compute_centroids_ras(fixed_seg_path_name, labels_registration)
    centroid_moving, valid_mask = compute_centroids_ras(moving_seg_path_name, labels_registration)
    affine_ras = getM(centroid_fixed[:, valid_mask > 0], centroid_moving[:, valid_mask > 0], use_L1=False)

    transform_matrix_template_to_mni = ras_to_lps_affine(affine_ras)
    apply_precomputed_rigid_registration_affine_matrix(
        fixed_img_path_name, moving_img_path_name, out_img_path_name, transform_matrix_template_to_mni,
        verify=verify, verbose=verbose,
    )
    if out_seg_path_name is not None:
        apply_precomputed_rigid_registration_affine_matrix(
            fixed_seg_path_name, moving_seg_path_name, out_seg_path_name, transform_matrix_template_to_mni,
            is_label=True, verify=verify, verbose=verbose,
        )

    if out_affine_matrix_path_name is not None:
        np.save(out_affine_matrix_path_name, transform_matrix_template_to_mni)


def uslr_mni_registration(
    moving_img_path_name: str,
    moving_seg_path_name: str,
    out_img_path_name: str,
    out_seg_path_name: str | None = None,
    out_affine_matrix_path_name: str | None = None,
    verify: bool = False,
    verbose: bool = False,
) -> None:
    """Convenience wrapper: register a moving image to the fixed MNI152 template.

    Equivalent to :func:`uslr_registration` with the fixed image/segmentation
    hardcoded to :data:`MNI_IMG_PATH` / :data:`MNI_SEG_PATH`. This is the
    function called by the main preprocessing pipeline
    (``src/preprocessing/preprocess_images.py``).

    Args:
        moving_img_path_name: Path to the moving (source) image.
        moving_seg_path_name: Path to the segmentation of the moving image.
        out_img_path_name: Path to write the MNI-registered image to.
        out_seg_path_name: Optional path to also write the segmentation
            resampled into MNI space.
        out_affine_matrix_path_name: Optional path to save the computed
            affine transform as a ``.npy`` file.
        verify: If True, skip processing when outputs already exist.
        verbose: If True, log progress/skip messages.
    """
    uslr_registration(
        MNI_IMG_PATH,
        MNI_SEG_PATH,
        moving_img_path_name,
        moving_seg_path_name,
        out_img_path_name,
        out_seg_path_name=out_seg_path_name,
        out_affine_matrix_path_name=out_affine_matrix_path_name,
        verify=verify,
        verbose=verbose,
    )