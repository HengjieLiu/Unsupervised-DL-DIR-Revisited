# feature_encoder/__init__.py

from .feature_encoder import FeatureEncoder
from .registries import FEATURE_ENCODER_REGISTRY, BLOCK_REGISTRY

## deepseek suggest
from . import encoders  # This line is added to trigger encoder registration
from . import building_blocks 

__all__ = ['FeatureEncoder', 'FEATURE_ENCODER_REGISTRY', 'BLOCK_REGISTRY']

