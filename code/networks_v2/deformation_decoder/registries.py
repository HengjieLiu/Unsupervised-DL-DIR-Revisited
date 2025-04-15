# deformation_decoder/registries.py

# Registry for top-level deformation decoder types.
DEFORMATION_DECODER_REGISTRY = {}

def register_deformation_decoder(decoder_type):
    def decorator(cls):
        DEFORMATION_DECODER_REGISTRY[decoder_type.lower()] = cls
        return cls
    return decorator

# Registry for building block types used in deformation decoders.
DEFORMATION_DECODER_BLOCK_REGISTRY = {}

def register_deformation_decoder_block(block_type):
    def decorator(cls):
        DEFORMATION_DECODER_BLOCK_REGISTRY[block_type.lower()] = cls
        return cls
    return decorator
