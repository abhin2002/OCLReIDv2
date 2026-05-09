# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseMemory
from .global_ocl_memory_manager import GlobalOCLMemoryManager
from .part_ocl_memory_manager import PartOCLMemoryManager
from . import name_match

__all__ = [
    'BaseMemory',
    'GlobalOCLMemoryManager',
    'PartOCLMemoryManager',
    'name_match',
]
