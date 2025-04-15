# deformation_decoder/decoders/__init__.py

from .deformation_decoder_mysitreg import DeformationDecoderMySITReg

from .deformation_decoder_pyramidal_cnn import DeformationDecoderPyramidalCNN
from .deformation_decoder_pyramidal_vfa import DeformationDecoderPyramidalVFA
# from .deformation_decoder_pyramidal_cnn_ic import DeformationDecoderPyramidalCNNIC





__all__ = [
    'DeformationDecoderMySITReg',
    'DeformationDecoderPyramidalCNN',
    'DeformationDecoderPyramidalVFA',
    # 'DeformationDecoderPyramidalCNNIC'
]
