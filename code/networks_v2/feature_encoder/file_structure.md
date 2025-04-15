my_project/
├── feature_encoder/
│   ├── __init__.py
│   ├── registries.py               # Contains registry dictionaries and decorators.
│   ├── feature_encoder.py          # Top-level FeatureEncoder class that uses the registry.
│   ├── encoders/
│   │   ├── __init__.py             # Exports FeatureEncoderCNN, FeatureEncoderEdgeFilter, FeatureEncoderDWT.
│   │   ├── feature_encoder_cnn.py
│   │   ├── feature_encoder_edge_filter.py
│   │   └── feature_encoder_dwt.py
│   └── building_blocks/
│       ├── __init__.py             # Exports building block classes.
│       ├── conv_block_with_skip.py
│       ├── depth_point_conv_block_with_skip.py
│       └── depth_point_conv_block_with_skip_channel_attention.py
└── requirements.txt
