"""SimRL - pointing / reaching in a swaying environment (MuJoCo + YOLO + PPO)."""

# ---------------------------------------------------------------------------
# This machine's anaconda numpy is linked against MKL *single-dynamic-library*
# and the OpenMP runtime (libiomp5md.dll) is missing from Library/bin, so the
# very first threaded BLAS kernel (any 2-D numpy matmul, np.dot, and therefore
# matplotlib) dies with a Windows delay-load error 0xC06D007E.  Forcing the
# sequential MKL layer makes everything work.  It has to be set *before* numpy
# is imported, hence it lives here and in every entry point.
# ---------------------------------------------------------------------------
import os as _os

_os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "1.0.0"
