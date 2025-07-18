#!/usr/bin/env python3
"""
D415 Multi-Modal Adaptive Detection System
Fraunhofer Refactored Version

A production-grade computer vision system for multi-modal object detection
using Intel RealSense D415 camera with RGB, Depth, IR, and Point Cloud data.
"""

__version__ = "1.0.0"
__author__ = "Fraunhofer Institute"

from .config import SystemConfig, ModalityType, load_config
from .core.detector import AdaptiveMultiModalDetector
from .main import DetectionApp, main

__all__ = [
    'SystemConfig',
    'ModalityType', 
    'load_config',
    'AdaptiveMultiModalDetector',
    'DetectionApp',
    'main'
]