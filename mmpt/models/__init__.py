# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401, F403
from .components import *
from .trackers  import *
from .builder import (build, build_backbone, build_loss,
                      build_model)
from .losses import *  # noqa: F401, F403
from .registry import BACKBONES, LOSSES, MODELS


# __all__ = [
#     'BaseModel', 'STM', 'build',
#     'build_backbone', 'build_loss', 'build_model', 'build_loss'
#     'BACKBONES', 'LOSSES'
# ]
