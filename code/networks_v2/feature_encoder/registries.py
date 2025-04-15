# feature_encoder/registries.py

# Registry for encoder types (e.g., 'cnn', 'edge_filter', 'dwt')
FEATURE_ENCODER_REGISTRY = {}

def register_encoder(encoder_type):
    def decorator(cls):
        FEATURE_ENCODER_REGISTRY[encoder_type.lower()] = cls
        return cls
    return decorator

# Registry for building blocks (e.g., 'conv', 'depth_point', 'depth_point_attention')
BLOCK_REGISTRY = {}

def register_block(block_type):
    def decorator(cls):
        BLOCK_REGISTRY[block_type.lower()] = cls
        return cls
    return decorator
