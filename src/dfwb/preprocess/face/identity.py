"""Finding a clip's main subject from the identity embeddings of the faces sampled from it.

This follows the earlier face pipeline's subject search: every face found on a handful of frames
spread across the clip is pooled, the pool is clustered by identity, and the cluster whose faces
are large in the frame, confidently detected and facing the camera (a small head yaw) wins. The
mean of its embeddings, scaled to unit length, is the subject that frame-by-frame selection then
looks for.

The clustering is average-linkage agglomerative clustering on cosine distance, cut at a distance
threshold: the same flat clusters as scipy's ``fcluster(linkage(pdist(x, "cosine"), "average"),
threshold, "distance")``, which the earlier pipeline called, but written in plain numpy so the
framework does not need scipy at runtime. Numpy itself is imported inside the functions, matching
the rest of the framework.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from dfwb.preprocess.face.types import Face

__all__ = ["cluster_subject"]

# Cosine distance runs from 0 (same direction) to 2 (opposite directions). A vector of all zeros
# has no direction, so its distance to anything is undefined; it gets the largest distance, which
# keeps it out of every cluster rather than stopping the clustering.
_UNDEFINED_DISTANCE = 2.0


def cluster_subject(
    faces: Sequence[Face], *, threshold: float = 0.5
) -> npt.NDArray[np.float32] | None:
    """The embedding of the clip's main subject, found by clustering ``faces`` by identity.

    Faces without an embedding are left out. The rest are clustered (average linkage on cosine
    distance, merging while the closest clusters are at most ``threshold`` apart), and each
    cluster is scored as the sum over its faces of ``box area * detection score * max(0, 1 -
    |yaw| / 90)``. Here ``yaw`` is ``Face.yaw``, the head's left-right turn in degrees, so a face
    counts for less the further it is turned towards profile, and for nothing from 90 degrees on;
    a face with no yaw estimate counts as facing the camera (a weight of 1). The highest-scoring
    cluster wins, ties going to the cluster whose first face comes first in ``faces``.

    Args:
        faces: The faces found on the frames sampled from the clip, in the order they were found.
        threshold: The cosine distance at which clusters stop merging.

    Returns:
        The mean of the winning cluster's embeddings, scaled to unit length, as ``float32``. With
        exactly one embedded face, that face's own embedding scaled to unit length. A vector too
        short to scale (length below 1e-12) is returned unscaled. ``None`` when no face has an
        embedding.
    """
    import numpy as np

    pool = [face for face in faces if face.embedding is not None]
    if not pool:
        return None
    embeddings = [np.asarray(face.embedding, dtype=np.float32) for face in pool]
    if len(pool) == 1:
        return _unit_norm(embeddings[0].astype(np.float32, copy=True))

    clusters = _cosine_average_linkage(
        np.vstack(embeddings).astype(np.float64), threshold=threshold
    )
    # max() keeps the first of several equally scored clusters
    subject = max(clusters, key=lambda members: _cluster_score([pool[i] for i in members]))
    mean = np.mean(np.vstack([embeddings[i] for i in subject]), axis=0).astype(np.float32)
    return _unit_norm(mean)


def _cluster_score(faces: Sequence[Face]) -> float:
    """Sum of box area * detection score * frontal weight (from head yaw) over a cluster."""
    total = 0.0
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
        yaw = 0.0 if face.yaw is None else float(abs(face.yaw))
        frontal = max(0.0, 1.0 - yaw / 90.0)
        total += area * float(face.score) * frontal
    return total


def _unit_norm(vector: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    import numpy as np

    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector
    return (vector / norm).astype(np.float32)


def _cosine_average_linkage(points: npt.NDArray[Any], *, threshold: float) -> list[list[int]]:
    """Average-linkage clusters of ``points`` (one per row) under cosine distance.

    Starting from one cluster per point, the two closest clusters are merged, again and again,
    while they are at most ``threshold`` apart; the distance between two clusters is the mean
    distance between their points. Merge distances never shrink under average linkage, so this
    gives exactly the flat clusters of cutting the full dendrogram at ``threshold``.

    Returns:
        The clusters as lists of row indices, each list ascending, and the lists ordered by their
        first index.
    """
    import numpy as np

    x = np.asarray(points, dtype=np.float64)
    count = x.shape[0]
    if count < 2:
        return [[index] for index in range(count)]

    norms = np.sqrt(np.einsum("ij,ij->i", x, x))
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = (x @ x.T) / np.outer(norms, norms)
    distance = 1.0 - np.clip(cosine, -1.0, 1.0)
    distance = np.nan_to_num(
        distance, nan=_UNDEFINED_DISTANCE, posinf=_UNDEFINED_DISTANCE, neginf=_UNDEFINED_DISTANCE
    )
    # a cluster is never merged with itself, nor with a cluster that no longer exists
    np.fill_diagonal(distance, np.inf)

    members = [[index] for index in range(count)]
    active = list(range(count))
    while len(active) > 1:
        between = distance[np.ix_(active, active)]
        row, column = divmod(int(np.argmin(between)), len(active))
        if between[row, column] > threshold:
            break
        keep, gone = sorted((active[row], active[column]))
        keep_size, gone_size = len(members[keep]), len(members[gone])
        merged = (keep_size * distance[keep] + gone_size * distance[gone]) / (keep_size + gone_size)
        distance[keep, :] = merged
        distance[:, keep] = merged
        distance[keep, keep] = np.inf
        distance[gone, :] = np.inf
        distance[:, gone] = np.inf
        members[keep] = sorted(members[keep] + members[gone])
        active.remove(gone)

    return [members[index] for index in active]
