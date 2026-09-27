"""The face pipeline: turns raw video into cropped face clips (contract C3c).

:mod:`~dfwb.preprocess.face.profiles` holds the named, hashed recipes (``ProcessingProfile``) and
the profiles shipped with dfwb. The backend contract, tracking, cropping and the store writer are
added alongside it as the pipeline grows. Nothing in this package imports cv2, PyAV or numpy at
module import time: those only load once a profile is actually run, so listing or validating
profiles never needs a media library installed.
"""

from __future__ import annotations
