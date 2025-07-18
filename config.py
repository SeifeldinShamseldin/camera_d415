#!/usr/bin/env python3
"""
Configuration module for D415 Multi-Modal Detection System
Centralizes all parameters, thresholds, and settings
"""

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass, field
from enum import Enum

class ModalityType(Enum):
    """Detection modality types"""
    RGB_SIFT = "rgb_sift"
    RGB_TEMPLATE = "rgb_template"
    IR_TEMPLATE = "ir_template"
    DEPTH_EDGE = "depth_edge"
    POINTCLOUD_ICP = "pointcloud_icp"
    CONTOUR_SHAPE = "contour_shape"
    DEPTH_HISTOGRAM = "depth_histogram"
    GEOMETRIC_PRIMITIVES = "geometric_primitives"
    SURFACE_NORMALS = "surface_normals"
    MULTI_MODAL = "multi_modal"

@dataclass
class CameraConfig:
    """Camera configuration parameters"""
    width: int = 640
    height: int = 480
    fps: int = 30
    
    # Default camera matrix (should be calibrated for actual camera)
    camera_matrix: np.ndarray = field(default_factory=lambda: np.array([
        [600.0, 0, 320.0],
        [0, 600.0, 240.0],
        [0, 0, 1.0]
    ], dtype=np.float32))
    
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))

@dataclass
class DetectionParams:
    """Detection algorithm parameters"""
    # SIFT parameters (original working values)
    sift_features: int = 300
    sift_ratio: float = 0.7  # Original working value
    min_inliers: int = 10    # Original working value
    
    # Template matching thresholds (original working values)
    template_threshold: float = 0.75  # Original working value
    ir_threshold: float = 0.65        # Original working value
    depth_edge_threshold: float = 0.6 # Original working value
    
    # ICP parameters
    icp_threshold: float = 0.8
    icp_voxel_size: float = 5.0
    icp_distance_threshold: float = 20.0
    
    # Object constraints
    min_area: int = 1000
    max_area: int = 50000
    
    # NMS parameters (original working values)
    nms_threshold: float = 0.5  # Original working value
    
    # Temporal stability parameters (disabled for now)
    min_detection_confidence: float = 0.0  # No filtering
    detection_stability_frames: int = 1    # Show immediately
    
    # Template analysis thresholds
    texture_sift_threshold: int = 20
    edge_density_threshold: float = 0.02
    depth_variance_threshold: float = 100.0
    ir_contrast_threshold: float = 0.1
    
    # Template matching scales
    template_scales: list = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])
    
    # Contour-based detection parameters (more sensitive)
    contour_area_threshold: int = 200  # Lowered from 500
    contour_match_threshold: float = 0.3  # Increased from 0.1 (less strict)
    hu_moments_threshold: float = 0.4     # Increased from 0.2 (less strict)
    
    # Depth histogram parameters
    depth_bins: int = 30  # Reduced for better matching
    histogram_correlation_threshold: float = 0.5  # Lowered from 0.7
    
    # Geometric primitives parameters
    hough_circle_param1: int = 50
    hough_circle_param2: int = 30
    hough_line_threshold: int = 100
    min_circle_radius: int = 10
    max_circle_radius: int = 200
    
    # Surface normal parameters
    normal_neighborhood_size: int = 5
    normal_correlation_threshold: float = 0.8

@dataclass
class PerformanceConfig:
    """Performance and optimization settings"""
    thread_pool_workers: int = 4  # Original working value
    detection_timeout: float = 0.1  # Original working value
    cache_timeout: float = 0.1  # seconds
    frame_skip: int = 0
    
    # ORB fallback parameters
    orb_features: int = 500

@dataclass
class UIConfig:
    """User interface configuration"""
    colors: Dict[ModalityType, tuple] = field(default_factory=lambda: {
        ModalityType.RGB_SIFT: (0, 255, 0),      # Green
        ModalityType.RGB_TEMPLATE: (255, 255, 0), # Yellow
        ModalityType.IR_TEMPLATE: (255, 0, 255),  # Magenta
        ModalityType.DEPTH_EDGE: (0, 255, 255),   # Cyan
        ModalityType.POINTCLOUD_ICP: (0, 165, 255), # Orange
        ModalityType.CONTOUR_SHAPE: (255, 165, 0), # Orange-red
        ModalityType.DEPTH_HISTOGRAM: (128, 0, 128), # Purple
        ModalityType.GEOMETRIC_PRIMITIVES: (255, 20, 147), # Deep pink
        ModalityType.SURFACE_NORMALS: (0, 255, 127), # Spring green
        ModalityType.MULTI_MODAL: (0, 255, 0)     # Green
    })
    
    # Overlay settings
    overlay_height: int = 120
    overlay_alpha: float = 0.7
    
    # Font settings
    font_scale: float = 0.6
    font_thickness: int = 2

@dataclass
class SystemConfig:
    """Overall system configuration"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionParams = field(default_factory=DetectionParams)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Paths
    template_dir: str = "multimodal_templates"
    
    # Feature flags
    enable_ir: bool = False  # Disabled due to D415 bandwidth limitations
    enable_pointcloud: bool = True
    enable_pose_estimation: bool = True
    
    # Featureless detection flags (disabled for stability like original)
    enable_contour_detection: bool = False
    enable_depth_histogram: bool = False
    enable_geometric_primitives: bool = False
    enable_surface_normals: bool = False
    
    # Logging
    verbose: bool = True
    log_detections: bool = False

# Default configuration instance
DEFAULT_CONFIG = SystemConfig()

def load_config(config_path: str = None) -> SystemConfig:
    """Load configuration from file or return default"""
    if config_path is None:
        return DEFAULT_CONFIG
    
    # TODO: Implement JSON/YAML config loading
    return DEFAULT_CONFIG

def save_config(config: SystemConfig, config_path: str):
    """Save configuration to file"""
    # TODO: Implement JSON/YAML config saving
    pass

# Modality weights for different object types
MODALITY_WEIGHTS = {
    'textured_objects': {
        ModalityType.RGB_SIFT: 0.5,
        ModalityType.RGB_TEMPLATE: 0.2,
        ModalityType.IR_TEMPLATE: 0.1,
        ModalityType.DEPTH_EDGE: 0.1,
        ModalityType.POINTCLOUD_ICP: 0.1
    },
    'featureless_objects': {
        ModalityType.RGB_TEMPLATE: 0.3,
        ModalityType.IR_TEMPLATE: 0.3,
        ModalityType.DEPTH_EDGE: 0.2,
        ModalityType.POINTCLOUD_ICP: 0.2
    },
    'reflective_objects': {
        ModalityType.IR_TEMPLATE: 0.4,
        ModalityType.DEPTH_EDGE: 0.3,
        ModalityType.POINTCLOUD_ICP: 0.3
    }
}