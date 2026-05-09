# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .global_ocl_classifier import GlobalOCLClassifier
from .global_r18_naive_classifier import GlobalResNetClassifier
from .part_ocl_weighted_classifier import PartOCLWeightedClassifier
from . import name_match

__all__ = [
    'BaseClassifier',
    'GlobalOCLClassifier',
    'GlobalResNetClassifier',
    'PartOCLWeightedClassifier',
    'name_match',
]
