"""ArcFace identity embeddings: the recognition model's input from an aligned face.

Ported from insightface 0.7.3 (https://github.com/deepinsight/insightface, MIT licence, copyright
Jiankang Deng and Jia Guo; see ``NOTICE``), ``python-package/insightface/model_zoo/
arcface_onnx.py``, class ``ArcFaceONNX``.

What differs from the original:

- :meth:`ArcFace.get_feat` receives aligned faces in red-green-blue order and builds the network
  input without swapping channels. insightface received blue-green-red and swapped, so from the
  same aligned face the network input is identical. The aligned face itself can differ by a grey
  level or two, because OpenCV 5's ``warpAffine`` rounds slightly differently from the OpenCV 4
  insightface ran with; the embedding then differs by at most about 0.002 per value (a cosine
  similarity of at least 0.9999), where the detections stay bit-identical.
- The input normalisation is passed in (``input_mean``, ``input_std``; 127.5 and 127.5 by
  default) instead of being read off the model graph, which needs the ``onnx`` package.
  insightface uses 0 and 1 for a model converted from MXNet, recognisable by ``Sub`` and ``Mul``
  nodes among its first eight, and 127.5 and 127.5 otherwise. The ``buffalo_l`` recognition model
  (``w600k_r50.onnx``) has neither node there (its first eight are convolution, PReLU, batch
  normalisation and addition nodes), so it takes the default.
- :meth:`ArcFace.get` takes the five landmarks and returns the embedding, rather than reading
  them from and writing the embedding onto insightface's ``Face`` object.
- The model is handed in as an open onnxruntime session; ``prepare()``, ``compute_sim()`` and
  ``forward()``, which the pipeline does not use, are left out. A model without exactly one output
  raises ``ValueError`` (the original asserted it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dfwb.preprocess.face._insightface_port.face_align import norm_crop

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["ArcFace"]


class ArcFace:
    """An ArcFace recognition model over an onnxruntime session.

    Args:
        session: An onnxruntime ``InferenceSession`` (or anything with its ``get_inputs``,
            ``get_outputs`` and ``run``) holding an ArcFace model.
        input_mean: Subtracted from every pixel value.
        input_std: Every pixel value is then divided by this.
    """

    def __init__(
        self, session: Any, *, input_mean: float = 127.5, input_std: float = 127.5
    ) -> None:
        self.session = session
        self.input_mean = input_mean
        self.input_std = input_std
        input_cfg = session.get_inputs()[0]
        input_shape = input_cfg.shape
        self.input_size: tuple[int, ...] = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = session.get_outputs()
        self.input_name = input_cfg.name
        self.output_names = [out.name for out in outputs]
        if len(self.output_names) != 1:
            raise ValueError(
                f"a recognition model has one output; this one has {len(self.output_names)}"
            )
        self.output_shape = outputs[0].shape

    def get(self, img: npt.NDArray[np.uint8], kps: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """The embedding of the face with five landmarks ``kps`` in the RGB frame ``img``, as the
        model returns it (not yet scaled to unit length)."""
        aimg = norm_crop(img, landmark=kps, image_size=self.input_size[0])
        embedding: npt.NDArray[Any] = self.get_feat(aimg).flatten()
        return embedding

    def get_feat(self, imgs: npt.NDArray[np.uint8] | list[npt.NDArray[np.uint8]]) -> Any:
        """The model's output for one aligned RGB face or a list of them."""
        import cv2

        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size
        # insightface passed swapRB=True here because it was given BGR; these images are RGB.
        blob = cv2.dnn.blobFromImages(
            imgs,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=False,
        )
        return self.session.run(self.output_names, {self.input_name: blob})[0]
