# Copyright (c) OpenMMLab. All rights reserved.
"""
ReID utility constants and functions for part-based feature handling.
"""

# Feature aggregation modes
GLOBAL = 'global'  # Global feature extraction
FOREGROUND = 'foreground'  # Foreground region features
CONCAT_PARTS = 'concat_parts'  # Concatenated part features
PARTS = 'parts'  # Part-based features (used as loss dict key)

__all__ = [
    'GLOBAL',
    'FOREGROUND', 
    'CONCAT_PARTS',
    'PARTS',
]
