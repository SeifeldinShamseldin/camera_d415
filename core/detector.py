#!/usr/bin/env python3
"""
Main detector class for D415 Multi-Modal Detection System
Coordinates all detection engines and manages the detection pipeline
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
from typing import List, Optional, Dict
import concurrent.futures

from ..config import SystemConfig, ModalityType
from ..detection.engines import (
    SIFTDetector, TemplateDetector, DepthEdgeDetector, 
    PointCloudICPDetector, DetectionFusion, Detection
)
from ..detection.featureless_engines import (
    ContourShapeDetector, DepthHistogramDetector,
    GeometricPrimitivesDetector, SurfaceNormalsDetector
)
from ..core.template_manager import TemplateManager
from ..utils.pose_estimation import PoseEstimator, PoseVisualizer, PoseFilter

class AdaptiveMultiModalDetector:
    """
    Production-grade multi-modal object detection system
    Adaptively selects best modality for each object type
    """
    
    def __init__(self, config: SystemConfig = None):
        # Load configuration
        from ..config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG
        
        # Camera configuration
        self.width = self.config.camera.width
        self.height = self.config.camera.height
        self.fps = self.config.camera.fps
        
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.pc = rs.pointcloud()
        self.align = None
        self.depth_intrinsics = None
        
        # Initialize detection engines
        self._initialize_detection_engines()
        
        # Initialize template manager
        self.template_manager = TemplateManager(
            self.config.template_dir, 
            self.config.detection
        )
        
        # Initialize pose estimation
        self.pose_estimator = PoseEstimator(
            self.config.camera.camera_matrix,
            self.config.camera.dist_coeffs
        )
        self.pose_visualizer = PoseVisualizer(self.pose_estimator)
        self.pose_filters = {}  # Per-template pose filters
        
        # Performance optimization
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.performance.thread_pool_workers
        )
        self.detection_cache = {}
        self.cache_timeout = self.config.performance.cache_timeout
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'modality_usage': {m: 0 for m in ModalityType},
            'avg_confidence': 0,
            'avg_time': 0
        }
        
        # Detection stability tracking
        self.detection_history = {}  # Track detections over time
        self.stable_detections = {}  # Currently stable detections
        self.last_detections = []    # Store last frame detections for smoothing
        
        if self.config.verbose:
            print("🚀 Multi-Modal Adaptive Detector Initialized")
    
    def _initialize_detection_engines(self):
        """Initialize all detection engines"""
        params = self.config.detection
        
        # Traditional detectors
        self.sift_detector = SIFTDetector(params)
        self.template_detector = TemplateDetector(params)
        self.depth_edge_detector = DepthEdgeDetector(params)
        self.pointcloud_icp_detector = None  # Initialized after camera setup
        
        # Featureless object detectors
        self.contour_detector = ContourShapeDetector(params)
        self.depth_histogram_detector = DepthHistogramDetector(params)
        self.geometric_primitives_detector = GeometricPrimitivesDetector(params)
        self.surface_normals_detector = None  # Initialized after camera setup
    
    def initialize_camera(self) -> bool:
        """Initialize D415 with working stream configuration"""
        try:
            # Configure streams
            self.rs_config.enable_stream(
                rs.stream.color, self.width, self.height, 
                rs.format.bgr8, self.fps
            )
            self.rs_config.enable_stream(
                rs.stream.depth, self.width, self.height, 
                rs.format.z16, self.fps
            )
            
            if self.config.enable_ir:
                # Enable IR streams if supported
                try:
                    self.rs_config.enable_stream(
                        rs.stream.infrared, 1, self.width, self.height, 
                        rs.format.y8, self.fps
                    )
                    self.rs_config.enable_stream(
                        rs.stream.infrared, 2, self.width, self.height, 
                        rs.format.y8, self.fps
                    )
                except:
                    if self.config.verbose:
                        print("⚠️  IR streams disabled due to bandwidth limitations")
                    self.config.enable_ir = False
            
            # Start pipeline
            profile = self.pipeline.start(self.rs_config)
            
            # Get device info and optimize settings
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            
            # Get intrinsics
            depth_profile = profile.get_stream(rs.stream.depth)
            color_profile = profile.get_stream(rs.stream.color)
            self.depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            # Update camera matrix with actual calibration
            self.config.camera.camera_matrix = np.array([
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1.0]
            ], dtype=np.float32)
            
            # Update pose estimator with real calibration
            self.pose_estimator = PoseEstimator(
                self.config.camera.camera_matrix,
                self.config.camera.dist_coeffs
            )
            self.pose_visualizer = PoseVisualizer(self.pose_estimator)
            
            # Initialize point cloud detector with intrinsics
            if self.config.enable_pointcloud:
                self.pointcloud_icp_detector = PointCloudICPDetector(
                    self.config.detection, self.depth_intrinsics
                )
            
            # Initialize surface normals detector with intrinsics
            self.surface_normals_detector = SurfaceNormalsDetector(
                self.config.detection, self.depth_intrinsics
            )
            
            # Create alignment object
            self.align = rs.align(rs.stream.color)
            
            if self.config.verbose:
                print("✅ D415 initialized:")
                print(f"   RGB: {self.width}x{self.height}@{self.fps}fps")
                print(f"   Depth: {self.width}x{self.height}@{self.fps}fps")
                if self.config.enable_ir:
                    print(f"   IR: {self.width}x{self.height}@{self.fps}fps")
                if self.config.enable_pointcloud:
                    print(f"   Point Cloud: Enabled")
            
            # Warm up
            for _ in range(10):
                self.pipeline.wait_for_frames()
                
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def create_template(self, name: str, rgb: np.ndarray, depth: np.ndarray, 
                       ir: Optional[np.ndarray], roi: tuple) -> bool:
        """Create a new template"""
        return self.template_manager.create_template(
            name, rgb, depth, ir, roi, self.depth_intrinsics
        )
    
    def load_templates(self):
        """Load all templates from disk"""
        self.template_manager.load_templates()
    
    def detect_adaptive(self, rgb: np.ndarray, depth: np.ndarray, 
                       ir: Optional[np.ndarray] = None) -> List[Detection]:
        """Adaptive detection using optimal modalities for each template"""
        all_detections = []
        
        # Get all templates
        templates = self.template_manager.get_all_templates()
        
        if not templates:
            return []
        
        # Process each template
        for template_name, template in templates.items():
            # Get template data for detection engines
            template_data = self.template_manager.get_template_data(template_name)
            if not template_data:
                continue
            
            # Get optimal modalities for this template
            modalities = template.analysis.optimal_modalities
            weights = template.analysis.weights
            
            # Run detection in parallel for each modality
            futures = []
            
            for modality in modalities:
                future = None
                
                if modality == ModalityType.RGB_SIFT:
                    future = self.thread_pool.submit(
                        self.sift_detector.detect, rgb, template_data
                    )
                elif modality == ModalityType.RGB_TEMPLATE:
                    future = self.thread_pool.submit(
                        self.template_detector.detect_rgb, rgb, template_data
                    )
                elif modality == ModalityType.IR_TEMPLATE and ir is not None:
                    future = self.thread_pool.submit(
                        self.template_detector.detect_ir, ir, template_data
                    )
                elif modality == ModalityType.DEPTH_EDGE:
                    future = self.thread_pool.submit(
                        self.depth_edge_detector.detect, depth, template_data
                    )
                elif (modality == ModalityType.POINTCLOUD_ICP and 
                      self.pointcloud_icp_detector is not None):
                    future = self.thread_pool.submit(
                        self.pointcloud_icp_detector.detect, depth, template_data
                    )
                elif (modality == ModalityType.CONTOUR_SHAPE and 
                      self.config.enable_contour_detection):
                    future = self.thread_pool.submit(
                        self.contour_detector.detect, rgb, template_data
                    )
                elif (modality == ModalityType.DEPTH_HISTOGRAM and 
                      self.config.enable_depth_histogram):
                    future = self.thread_pool.submit(
                        self.depth_histogram_detector.detect, depth, template_data
                    )
                elif (modality == ModalityType.GEOMETRIC_PRIMITIVES and 
                      self.config.enable_geometric_primitives):
                    future = self.thread_pool.submit(
                        self.geometric_primitives_detector.detect, rgb, template_data
                    )
                elif (modality == ModalityType.SURFACE_NORMALS and 
                      self.surface_normals_detector is not None and
                      self.config.enable_surface_normals):
                    future = self.thread_pool.submit(
                        self.surface_normals_detector.detect, depth, template_data
                    )
                
                if future:
                    futures.append((modality, future))
            
            # Collect results
            modality_detections = []
            for modality, future in futures:
                try:
                    result = future.result(timeout=self.config.performance.detection_timeout)
                    if result:
                        modality_detections.append(result)
                except:
                    pass
            
            # Fuse detections if multiple modalities detected
            if len(modality_detections) > 1:
                fused = DetectionFusion.fuse_detections(modality_detections, weights)
                if fused:
                    all_detections.append(fused)
            elif len(modality_detections) == 1:
                all_detections.append(modality_detections[0])
        
        # Apply NMS (like original code)
        final_detections = self._apply_nms(all_detections)
        
        # Enhance detections with pose estimation
        enhanced_detections = []
        for detection in final_detections:
            enhanced_detection = self._enhance_detection_with_pose(detection)
            enhanced_detections.append(enhanced_detection)
        
        # Update statistics
        for det in enhanced_detections:
            self.stats['total_detections'] += 1
            self.stats['modality_usage'][det.modality] += 1
        
        return enhanced_detections
    
    def _apply_temporal_smoothing(self, current_detections: List[Detection]) -> List[Detection]:
        """Apply temporal smoothing to reduce detection flickering"""
        # If no previous detections, just return current
        if not self.last_detections:
            self.last_detections = current_detections
            return current_detections
        
        # Simple smoothing: if we had detections last frame and this frame, keep them
        # This prevents flickering where objects appear/disappear rapidly
        smoothed = []
        
        # Add all current detections (they passed confidence threshold)
        smoothed.extend(current_detections)
        
        # Also add previous detections that are close to current ones (prevents disappearing)
        for prev_det in self.last_detections:
            found_similar = False
            for curr_det in current_detections:
                if (prev_det.template_name == curr_det.template_name and
                    self._detections_close(prev_det, curr_det)):
                    found_similar = True
                    break
            
            # If previous detection is close but not found, add it with reduced confidence
            if not found_similar:
                # Reduce confidence slightly and add
                prev_det.confidence *= 0.9
                if prev_det.confidence > 0.05:  # Still above minimum
                    smoothed.append(prev_det)
        
        # Update last detections
        self.last_detections = current_detections
        
        return smoothed
    
    def _detections_close(self, det1: Detection, det2: Detection) -> bool:
        """Check if two detections are spatially close"""
        try:
            # Get center points
            center1 = np.mean(det1.corners.reshape(-1, 2), axis=0)
            center2 = np.mean(det2.corners.reshape(-1, 2), axis=0)
            
            # Check if centers are within 50 pixels
            distance = np.linalg.norm(center1 - center2)
            return distance < 50.0
        except:
            return False
    
    def _enhance_detection_with_pose(self, detection: Detection) -> Detection:
        """Enhance detection with 6D pose estimation"""
        # Skip if pose already estimated (e.g., from ICP)
        if detection.pose_6d.get('success', False):
            return detection
        
        # Estimate pose from 2D corners
        corners_2d = detection.corners.reshape(-1, 2)
        pose_6d = self.pose_estimator.estimate_pose_from_corners(corners_2d)
        
        # Debug pose estimation (uncomment for debugging)
        # if pose_6d.get('success', False):
        #     print(f"✅ 6D Pose: {detection.template_name} pos={pose_6d['position'][:3]}")
        
        # Apply temporal filtering if enabled
        if self.config.enable_pose_estimation:
            template_name = detection.template_name
            if template_name not in self.pose_filters:
                self.pose_filters[template_name] = PoseFilter()
            
            pose_6d = self.pose_filters[template_name].filter_pose(pose_6d)
        
        # Update detection with pose
        detection.pose_6d = pose_6d
        return detection
    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        keep = []
        for det1 in detections:
            should_keep = True
            
            for det2 in keep:
                # Calculate IoU
                iou = self._calculate_iou(
                    det1.corners.reshape(-1, 2),
                    det2.corners.reshape(-1, 2)
                )
                
                if iou > self.config.detection.nms_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(det1)
        
        return keep
    
    def _calculate_iou(self, corners1: np.ndarray, corners2: np.ndarray) -> float:
        """Calculate Intersection over Union"""
        try:
            # Create masks
            mask1 = np.zeros((self.height, self.width), dtype=np.uint8)
            mask2 = np.zeros((self.height, self.width), dtype=np.uint8)
            
            cv2.fillPoly(mask1, [corners1.astype(int)], 255)
            cv2.fillPoly(mask2, [corners2.astype(int)], 255)
            
            # Calculate intersection and union
            intersection = cv2.bitwise_and(mask1, mask2)
            union = cv2.bitwise_or(mask1, mask2)
            
            intersection_area = np.sum(intersection > 0)
            union_area = np.sum(union > 0)
            
            return intersection_area / union_area if union_area > 0 else 0.0
            
        except:
            return 0.0
    
    def draw_detection(self, frame: np.ndarray, detection: Detection):
        """Draw detection with 6D pose"""
        # Draw bounding box
        corners = detection.corners.astype(int).reshape(-1, 2)
        color = self.config.ui.colors.get(detection.modality, (255, 255, 255))
        
        # Draw polygon
        cv2.polylines(frame, [corners], True, color, 3)
        
        # Draw corners
        for corner in corners:
            cv2.circle(frame, tuple(corner), 5, color, -1)
        
        # Draw 6D pose if available
        if self.config.enable_pose_estimation and detection.pose_6d.get('success', False):
            self.pose_visualizer.draw_pose_axes(frame, detection.pose_6d)
        
        # Draw label and info
        center = np.mean(corners, axis=0).astype(int)
        
        # Template name and modality
        label = f"{detection.template_name} ({detection.modality.value})"
        cv2.putText(frame, label, (center[0] - 50, center[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.ui.font_scale, color, 
                   self.config.ui.font_thickness)
        
        # Confidence
        conf_text = f"{detection.confidence:.0%}"
        cv2.putText(frame, conf_text, (center[0] - 20, center[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 6D pose info
        if self.config.enable_pose_estimation and detection.pose_6d.get('success', False):
            self.pose_visualizer.draw_pose_info(
                frame, detection.pose_6d, (center[0] - 60, center[1] + 20), color
            )
    
    def draw_status_overlay(self, frame: np.ndarray, fps: float, detections: List[Detection]):
        """Draw status overlay with performance metrics"""
        overlay_height = self.config.ui.overlay_height
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
        alpha = self.config.ui.overlay_alpha
        frame[:overlay_height] = cv2.addWeighted(
            frame[:overlay_height], 1-alpha, overlay[:overlay_height], alpha, 0
        )
        
        # Title
        cv2.putText(frame, "D415 Multi-Modal Adaptive Detection", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Stats
        template_count = len(self.template_manager.get_all_templates())
        cv2.putText(frame, f"FPS: {fps:.1f} | Templates: {template_count} | Detections: {len(detections)}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Modality usage
        if detections:
            modalities = [d.modality.value for d in detections]
            modality_text = f"Active: {', '.join(set(modalities))}"
            cv2.putText(frame, modality_text, (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Controls
        cv2.putText(frame, "Controls: [t]emplate [s]ave [r]eload [q]uit", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.pipeline.stop()
            self.thread_pool.shutdown(wait=True)
        except:
            pass
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return self.stats.copy()