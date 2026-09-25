"""A port of the pre- and post-processing that insightface 0.7.3 wraps around its ONNX models.

insightface (https://github.com/deepinsight/insightface, MIT licence, copyright Jiankang Deng and
Jia Guo) runs its ``buffalo_l`` detection and recognition models with onnxruntime and does
everything else in numpy and OpenCV. That everything-else is carried over here line for line, so
the same models give the same results without the insightface package itself: detections bit for
bit, and identity embeddings to within a small tolerance (see below).

- :mod:`.scrfd`: turning a frame into the detector's input, and its outputs into boxes, scores and
  five landmarks per face;
- :mod:`.face_align`: aligning a face to the recognition model's landmark template, with the
  similarity estimate insightface takes from scikit-image (BSD-3-Clause) ported alongside;
- :mod:`.arcface`: turning an aligned face into the recognition model's input.

The licences of both upstream projects are reproduced in the ``NOTICE`` file next to this module.

Embeddings are not bit for bit because of OpenCV, not the port: a face is aligned with OpenCV's
``warpAffine`` before it is embedded, and OpenCV 5 rounds that warp slightly differently from the
OpenCV 4 insightface 0.7.3 ran with (with OpenCV 4 the aligned face comes out identical). The
embedding then differs by at most about 0.002 per value, a cosine similarity of at least 0.9999.
Only identity-guided subject selection uses embeddings, and grouping faces by identity does not
turn on differences that small.

Two kinds of change were made on purpose. Frames arrive in red-green-blue order rather than
OpenCV's blue-green-red, so the channel swap insightface made while building each network input
is skipped, which leaves that input exactly as it was. And code the ``buffalo_l`` models never
reach is left out; each module lists what. numpy and OpenCV are imported inside the functions that
use them, as everywhere else in the framework.
"""

from __future__ import annotations
