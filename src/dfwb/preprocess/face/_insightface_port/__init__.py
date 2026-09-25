"""A port of the pre- and post-processing that insightface 0.7.3 wraps around its ONNX models.

insightface (https://github.com/deepinsight/insightface, MIT licence, copyright Jiankang Deng and
Jia Guo) runs its ``buffalo_l`` detection and recognition models with onnxruntime and does
everything else in numpy and OpenCV. That everything-else is carried over here line for line, so
the same models give the same numbers without the insightface package itself:

- :mod:`.scrfd`: turning a frame into the detector's input, and its outputs into boxes, scores and
  five landmarks per face;
- :mod:`.face_align`: aligning a face to the recognition model's landmark template, with the
  similarity estimate insightface takes from scikit-image (BSD-3-Clause) ported alongside;
- :mod:`.arcface`: turning an aligned face into the recognition model's input.

The licences of both upstream projects are reproduced in the ``NOTICE`` file next to this module.

Two kinds of change were made on purpose. Frames arrive in red-green-blue order rather than
OpenCV's blue-green-red, so the channel swap insightface made while building each network input
is skipped, which leaves that input exactly as it was. And code the ``buffalo_l`` models never
reach is left out; each module lists what. numpy and OpenCV are imported inside the functions that
use them, as everywhere else in the framework.
"""

from __future__ import annotations
