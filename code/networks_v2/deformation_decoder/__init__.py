# deformation_decoder/__init__.py

from .deformation_decoder import DeformationDecoder
from .registries import DEFORMATION_DECODER_REGISTRY, DEFORMATION_DECODER_BLOCK_REGISTRY

# Import subpackages to trigger registrations.
from . import decoders
from . import building_blocks

__all__ = ['DeformationDecoder', 'DEFORMATION_DECODER_REGISTRY', 'DEFORMATION_DECODER_BLOCK_REGISTRY']
