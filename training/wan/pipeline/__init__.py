
from .pipeline_wan2_2_fun_inpaint import Wan2_2FunInpaintPipeline
from .pipeline_wan2_2_fun_inpaint_hand_concat import Wan2_2FunInpaintHandConcatPipeline

Wan2_2I2VPipeline = Wan2_2FunInpaintPipeline

import importlib.util

if importlib.util.find_spec("pai_fuser") is not None:
    from pai_fuser.core import sparse_reset

    # Wan2.2
    Wan2_2FunInpaintPipeline.__call__ = sparse_reset(Wan2_2FunInpaintPipeline.__call__)
    Wan2_2I2VPipeline.__call__ = sparse_reset(Wan2_2I2VPipeline.__call__)
