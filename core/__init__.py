#!/usr/bin/env python3
"""
Core modules for the D415 Multi-Modal Detection System
"""

from .detector import AdaptiveMultiModalDetector
from .template_manager import TemplateManager, MultiModalTemplate, TemplateAnalysis

__all__ = [
    'AdaptiveMultiModalDetector',
    'TemplateManager', 
    'MultiModalTemplate',
    'TemplateAnalysis'
]