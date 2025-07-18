#!/usr/bin/env python3
"""
Detection engines for the D415 Multi-Modal Detection System
"""

from .engines import (
    Detection,
    SIFTDetector,
    TemplateDetector, 
    DepthEdgeDetector,
    PointCloudICPDetector,
    DetectionFusion
)

from .featureless_engines import (
    ContourShapeDetector,
    DepthHistogramDetector,
    GeometricPrimitivesDetector,
    SurfaceNormalsDetector
)

__all__ = [
    'Detection',
    'SIFTDetector',
    'TemplateDetector',
    'DepthEdgeDetector', 
    'PointCloudICPDetector',
    'DetectionFusion',
    'ContourShapeDetector',
    'DepthHistogramDetector',
    'GeometricPrimitivesDetector',
    'SurfaceNormalsDetector'
]