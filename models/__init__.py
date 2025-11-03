"""Model architectures for multi-task learning."""

from models.base import (
    MultiTaskModel,
    FeatureExtractor,
    CRNNBackbone,
    AECHead,
    VADHead,
)
from models.shared_backbone import SharedBackboneModel
from models.hierarchical import HierarchicalModel

__all__ = [
    'MultiTaskModel',
    'FeatureExtractor',
    'CRNNBackbone',
    'AECHead',
    'VADHead',
    'SharedBackboneModel',
    'HierarchicalModel',
]
