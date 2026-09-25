"""Aligning a face to the ArcFace landmark template before it is embedded.

Ported from insightface 0.7.3 (https://github.com/deepinsight/insightface, MIT licence, copyright
Jiankang Deng and Jia Guo; see ``NOTICE``), ``python-package/insightface/utils/face_align.py``:
``arcface_dst``, ``estimate_norm`` and ``norm_crop``. insightface estimates the similarity
transform with scikit-image's ``SimilarityTransform.estimate``; the function that does the work,
scikit-image 0.24.0's ``skimage/transform/_geometric.py:_umeyama`` (BSD-3-Clause, copyright the
scikit-image team; see ``NOTICE``), is ported here as :func:`umeyama` so scikit-image is not
needed.

What differs from the original:

- ``arcface_dst`` is kept as plain numbers and turned into the same ``float32`` array when used,
  so importing this module does not import numpy.
- ``estimate_norm``'s two ``assert`` statements raise ``ValueError`` instead (an ``assert`` is
  skipped when Python runs with ``-O``), and the unused ``mode`` argument is dropped.
- The functions ``norm_crop2``, ``square_crop``, ``transform`` and ``trans_points*``, which the
  recognition model does not use, are left out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["arcface_dst", "estimate_norm", "norm_crop", "umeyama"]

# Where ArcFace expects the five landmarks (left eye, right eye, nose tip, left and right mouth
# corners) in a 112 x 112 aligned face.
arcface_dst: tuple[tuple[float, float], ...] = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


def umeyama(src: Any, dst: Any, estimate_scale: bool) -> npt.NDArray[np.float64]:
    """Estimate an N-D similarity transformation with or without scaling.

    Umeyama's least-squares method ("Least-squares estimation of transformation parameters
    between two point patterns", Shinji Umeyama, PAMI 1991, doi:10.1109/34.88573), as
    scikit-image computes it.

    Args:
        src: ``(M, N)`` source coordinates.
        dst: ``(M, N)`` destination coordinates.
        estimate_scale: Whether to estimate the scaling factor.

    Returns:
        The ``(N + 1, N + 1)`` homogeneous similarity transformation matrix. It contains NaN
        values only if the problem is not well-conditioned.
    """
    import numpy as np

    src = np.asarray(src)
    dst = np.asarray(dst)

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        nan: npt.NDArray[np.float64] = np.nan * T
        return nan
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:  # noqa: SIM108 - kept as scikit-image wrote it
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


def estimate_norm(lmk: npt.NDArray[Any], image_size: int = 112) -> npt.NDArray[np.float64]:
    """The ``2 x 3`` affine matrix taking the five landmarks ``lmk`` onto the ArcFace template
    for an ``image_size`` x ``image_size`` aligned face.

    Raises:
        ValueError: ``lmk`` is not five ``(x, y)`` points, or ``image_size`` is not a multiple of
            112 or 128.
    """
    import numpy as np

    if lmk.shape != (5, 2):
        raise ValueError(f"expected five (x, y) landmarks, got an array of shape {lmk.shape}")
    if not (image_size % 112 == 0 or image_size % 128 == 0):
        raise ValueError(f"image_size must be a multiple of 112 or 128, got {image_size}")
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = np.array(arcface_dst, dtype=np.float32) * ratio
    dst[:, 0] += diff_x
    M: npt.NDArray[np.float64] = umeyama(lmk, dst, True)[0:2, :]
    return M


def norm_crop(
    img: npt.NDArray[np.uint8], landmark: npt.NDArray[Any], image_size: int = 112
) -> npt.NDArray[Any]:
    """``img`` warped so that the five landmarks ``landmark`` land on the ArcFace template,
    cropped to ``image_size`` x ``image_size``; outside the frame is black."""
    import cv2

    M = estimate_norm(landmark, image_size)
    warped: npt.NDArray[Any] = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped
