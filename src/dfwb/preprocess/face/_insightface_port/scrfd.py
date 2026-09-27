"""SCRFD face detection: the detector's input from a frame, and faces from its outputs.

Ported from insightface 0.7.3 (https://github.com/deepinsight/insightface, MIT licence, copyright
Jiankang Deng and Jia Guo; see ``NOTICE``), ``python-package/insightface/model_zoo/retinaface.py``.
That file's ``RetinaFace`` class is the one insightface's model router loads ``det_10g.onnx``
with. It is ``model_zoo/scrfd.py``'s ``SCRFD`` class minus one branch, for models whose outputs
carry a batch dimension; ``det_10g.onnx``'s outputs do not, so both classes compute the same
thing for it.

What differs from the original:

- :meth:`SCRFD.forward` receives the canvas in red-green-blue order and builds the network input
  without swapping channels. insightface received it in blue-green-red and swapped, so the
  network input is identical.
- The model is handed in as an open onnxruntime session, and ``prepare()``'s settings become
  constructor arguments. A model with a fixed input size is logged rather than printed, and an
  output layout the original does not know raises ``ValueError`` (the original failed later, with
  an ``AttributeError``).
- The anchor-centre grid (:func:`anchor_centers`) and suppression (:func:`nms`) are functions of
  their own, so each can be tested alone.
- Unused code is left out: ``softmax``, the torch-only ``max_shape`` clamping in
  :func:`distance2bbox` and :func:`distance2kps`, the model file loading and the module's demo.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["SCRFD", "anchor_centers", "distance2bbox", "distance2kps", "nms"]

_log = logging.getLogger(__name__)

# The anchor-centre grids kept per detector; insightface stops caching after this many.
_CENTER_CACHE_LIMIT = 100


def distance2bbox(points: npt.NDArray[Any], distance: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Decode distance predictions to bounding boxes.

    Args:
        points: Shape ``(n, 2)``, the anchor centres ``[x, y]``.
        distance: Shape ``(n, 4)``, the distance from each centre to the box's left, top, right
            and bottom edges.

    Returns:
        Shape ``(n, 4)``, the boxes ``[x1, y1, x2, y2]``.
    """
    import numpy as np

    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: npt.NDArray[Any], distance: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Decode offset predictions to landmarks.

    Args:
        points: Shape ``(n, 2)``, the anchor centres ``[x, y]``.
        distance: Shape ``(n, 2k)``, each landmark's ``(dx, dy)`` offset from the centre.

    Returns:
        Shape ``(n, 2k)``, the landmarks ``[x0, y0, x1, y1, ...]``.
    """
    import numpy as np

    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def anchor_centers(
    height: int, width: int, stride: int, num_anchors: int
) -> npt.NDArray[np.float32]:
    """The anchor centres of a ``height`` x ``width`` feature map with the given stride.

    Cells are taken row by row, each centre in input pixels (``x * stride, y * stride``), and each
    repeated ``num_anchors`` times in a row, matching the order of the detector's outputs.
    """
    import numpy as np

    grid = np.mgrid[:height, :width][::-1]
    centers: npt.NDArray[np.float32] = np.stack(grid, axis=-1).astype(np.float32)  # type: ignore[call-overload]
    centers = (centers * stride).reshape((-1, 2))
    if num_anchors > 1:
        centers = np.stack([centers] * num_anchors, axis=1).reshape((-1, 2))
    return centers


def nms(dets: npt.NDArray[Any], thresh: float) -> list[int]:
    """Greedy non-maximum suppression, as insightface's ``SCRFD.nms``.

    Boxes are visited best score first; each kept box removes every remaining box that overlaps
    it by more than ``thresh`` (intersection over union, counting pixels inclusively, so a box is
    ``x2 - x1 + 1`` pixels wide).

    Args:
        dets: Shape ``(n, 5)``, rows of ``[x1, y1, x2, y2, score]``.
        thresh: The overlap above which a box is suppressed.

    Returns:
        The row indices kept, best score first.
    """
    import numpy as np

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class SCRFD:
    """An SCRFD face detector over an onnxruntime session.

    Args:
        session: An onnxruntime ``InferenceSession`` (or anything with its ``get_inputs``,
            ``get_outputs`` and ``run``) holding an SCRFD model.
        input_size: The ``(width, height)`` of the network input, used unless the model fixes
            its own.
        det_thresh: Faces scoring below this are dropped.
        nms_thresh: The overlap above which a lower-scoring face is suppressed.
    """

    input_mean = 127.5
    input_std = 128.0

    def __init__(
        self,
        session: Any,
        *,
        input_size: tuple[int, int],
        det_thresh: float = 0.5,
        nms_thresh: float = 0.4,
    ) -> None:
        self.session = session
        self.center_cache: dict[tuple[int, int, int], npt.NDArray[np.float32]] = {}
        self.nms_thresh = nms_thresh
        self.det_thresh = det_thresh
        self._init_vars()
        if self._model_input_size is not None:
            _log.warning(
                "the detection model has a fixed input size %s; the requested %s is ignored",
                self._model_input_size,
                input_size,
            )
            self.input_size = self._model_input_size
        else:
            self.input_size = input_size

    def _init_vars(self) -> None:
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        self._model_input_size: tuple[int, int] | None
        if isinstance(input_shape[2], str):
            self._model_input_size = None
        else:
            self._model_input_size = tuple(input_shape[2:4][::-1])
        self.input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True
        else:
            raise ValueError(
                f"an SCRFD model has 6, 9, 10 or 15 outputs; this one has {len(outputs)} outputs"
            )

    def forward(
        self, img: npt.NDArray[np.uint8], threshold: float
    ) -> tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any]], list[npt.NDArray[Any]]]:
        """Run the network on the canvas ``img`` (RGB) and keep what scores at least
        ``threshold``, per feature map: scores, boxes and landmarks in canvas pixels."""
        import cv2
        import numpy as np

        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        # insightface passed swapRB=True here because it was given BGR; this image is RGB already.
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=False,
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                centers = self.center_cache[key]
            else:
                centers = anchor_centers(height, width, stride, self._num_anchors)
                if len(self.center_cache) < _CENTER_CACHE_LIMIT:
                    self.center_cache[key] = centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(
        self,
        img: npt.NDArray[np.uint8],
        input_size: tuple[int, int] | None = None,
        max_num: int = 0,
        metric: str = "default",
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[Any] | None]:
        """Detect the faces in the RGB frame ``img``.

        The frame is resized, keeping its aspect ratio, to fit the network input and placed in
        its top-left corner; what the network finds is scaled back to frame pixels, and
        overlapping detections are suppressed.

        Args:
            img: One ``[H, W, 3]`` ``uint8`` frame in RGB order.
            input_size: The network input ``(width, height)``; the detector's own by default.
            max_num: When above zero, keep at most this many faces, preferring large faces near
                the frame's centre (or, with ``metric="max"``, just large ones).
            metric: ``"default"`` or ``"max"``; see ``max_num``.

        Returns:
            ``(det, kpss)``: ``det`` of shape ``(n, 5)``, rows of ``[x1, y1, x2, y2, score]``,
            best score first; ``kpss`` of shape ``(n, 5, 2)``, each face's five landmarks, or
            ``None`` for a model that does not predict landmarks.
        """
        import cv2
        import numpy as np

        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss: Any = None
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = nms(pre_det, self.nms_thresh)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":  # noqa: SIM108 - kept as insightface wrote it
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss
