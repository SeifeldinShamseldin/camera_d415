#!/usr/bin/env python3
"""
Detection engines module for D415 Multi-Modal Detection System
Contains all detection algorithm implementations
"""

import cv2
import numpy as np
import time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

from ..config import ModalityType, DetectionParams

@dataclass
class Detection:
    """Detection result structure"""
    template_name: str
    confidence: float
    corners: np.ndarray
    pose_6d: Dict
    modality: ModalityType
    processing_time: float

class SIFTDetector:
    """SIFT-based detection for textured objects"""
    
    def __init__(self, params: DetectionParams):
        self.params = params
        self.detector = cv2.SIFT_create(nfeatures=params.sift_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
    
    def detect(self, rgb: np.ndarray, template_data: Dict) -> Optional[Detection]:
        """Perform SIFT-based detection"""
        if not template_data.get('sift_features') or template_data['sift_features']['descriptors'] is None:
            return None
            
        try:
            start_time = time.time()
            
            # Detect features in current frame
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            kp, desc = self.detector.detectAndCompute(gray, None)
            
            if desc is None or len(kp) < 10:
                return None
            
            # Match features
            matches = self.matcher.knnMatch(
                template_data['sift_features']['descriptors'], desc, k=2
            )
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.params.sift_ratio * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < self.params.min_inliers:
                return None
            
            # Find homography
            src_pts = np.float32([
                template_data['sift_features']['keypoints'][m.queryIdx].pt 
                for m in good_matches
            ]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                kp[m.trainIdx].pt for m in good_matches
            ]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                return None
            
            # Transform template corners
            h, w = template_data['rgb'].shape[:2]
            corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, H)
            
            # Validate detection
            area = cv2.contourArea(transformed_corners)
            if area < self.params.min_area or area > self.params.max_area:
                return None
            
            # Calculate confidence
            inliers = np.sum(mask)
            confidence = min(1.0, inliers / len(good_matches) * (area / 10000))
            
            processing_time = time.time() - start_time
            
            return Detection(
                template_name=template_data['name'],
                confidence=confidence,
                corners=transformed_corners,
                pose_6d={'success': False},  # Will be calculated by pose estimator
                modality=ModalityType.RGB_SIFT,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"SIFT detection error: {e}")
            return None

class TemplateDetector:
    """Traditional template matching for featureless objects"""
    
    def __init__(self, params: DetectionParams):
        self.params = params
    
    def detect_rgb(self, rgb: np.ndarray, template_data: Dict) -> Optional[Detection]:
        """RGB template matching"""
        try:
            start_time = time.time()
            
            # Convert to grayscale
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template_data['rgb'], cv2.COLOR_BGR2GRAY)
            
            best_match = self._multi_scale_match(gray, template_gray, self.params.template_threshold)
            
            if best_match is None:
                return None
            
            processing_time = time.time() - start_time
            
            return Detection(
                template_name=template_data['name'],
                confidence=best_match['confidence'],
                corners=best_match['corners'],
                pose_6d={'success': False},
                modality=ModalityType.RGB_TEMPLATE,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"RGB template matching error: {e}")
            return None
    
    def detect_ir(self, ir: np.ndarray, template_data: Dict) -> Optional[Detection]:
        """IR template matching for smooth/reflective objects"""
        if template_data['ir'] is None:
            return None
            
        try:
            start_time = time.time()
            
            # Enhance IR images
            ir_enhanced = cv2.equalizeHist(ir)
            template_ir_enhanced = cv2.equalizeHist(template_data['ir'])
            
            best_match = self._multi_scale_match(
                ir_enhanced, template_ir_enhanced, 
                self.params.ir_threshold,
                methods=[cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
            )
            
            if best_match is None:
                return None
            
            processing_time = time.time() - start_time
            
            return Detection(
                template_name=template_data['name'],
                confidence=best_match['confidence'],
                corners=best_match['corners'],
                pose_6d={'success': False},
                modality=ModalityType.IR_TEMPLATE,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"IR detection error: {e}")
            return None
    
    def _multi_scale_match(self, image: np.ndarray, template: np.ndarray, 
                          threshold: float, methods: List = None) -> Optional[Dict]:
        """Perform multi-scale template matching"""
        if methods is None:
            methods = [cv2.TM_CCOEFF_NORMED]
        
        best_match = None
        best_val = 0
        
        for scale in self.params.template_scales:
            scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
            
            if (scaled_template.shape[0] > image.shape[0] or 
                scaled_template.shape[1] > image.shape[1]):
                continue
            
            for method in methods:
                result = cv2.matchTemplate(image, scaled_template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_val and max_val > threshold:
                    best_val = max_val
                    h, w = scaled_template.shape
                    x, y = max_loc
                    
                    corners = np.array([
                        [x, y], [x+w, y], [x+w, y+h], [x, y+h]
                    ], dtype=np.float32).reshape(-1, 1, 2)
                    
                    best_match = {
                        'corners': corners,
                        'confidence': max_val,
                        'scale': scale
                    }
        
        return best_match

class DepthEdgeDetector:
    """Depth edge-based detection"""
    
    def __init__(self, params: DetectionParams):
        self.params = params
    
    def detect(self, depth: np.ndarray, template_data: Dict) -> Optional[Detection]:
        """Depth edge-based detection"""
        try:
            start_time = time.time()
            
            # Convert depth to edge map
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_edges = cv2.Canny(depth_norm, 50, 150)
            
            template_depth_norm = cv2.normalize(
                template_data['depth'], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            template_edges = cv2.Canny(template_depth_norm, 50, 150)
            
            # Template matching on edges
            result = cv2.matchTemplate(depth_edges, template_edges, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val < self.params.depth_edge_threshold:
                return None
            
            h, w = template_edges.shape
            x, y = max_loc
            
            corners = np.array([
                [x, y], [x+w, y], [x+w, y+h], [x, y+h]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            processing_time = time.time() - start_time
            
            return Detection(
                template_name=template_data['name'],
                confidence=max_val,
                corners=corners,
                pose_6d={'success': False},
                modality=ModalityType.DEPTH_EDGE,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"Depth edge detection error: {e}")
            return None

class PointCloudICPDetector:
    """Point cloud ICP-based detection"""
    
    def __init__(self, params: DetectionParams, depth_intrinsics):
        self.params = params
        self.depth_intrinsics = depth_intrinsics
    
    def detect(self, depth: np.ndarray, template_data: Dict) -> Optional[Detection]:
        """Point cloud ICP-based detection"""
        if not OPEN3D_AVAILABLE or template_data.get('pointcloud') is None:
            return None
            
        try:
            start_time = time.time()
            
            # Convert current depth to point cloud
            current_pc = self._depth_to_pointcloud(depth)
            
            if len(current_pc) < 100:
                return None
            
            # Create Open3D point clouds
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(template_data['pointcloud'])
            
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(current_pc)
            
            # Downsample for performance
            source = source.voxel_down_sample(self.params.icp_voxel_size)
            target = target.voxel_down_sample(self.params.icp_voxel_size)
            
            # Estimate normals
            source.estimate_normals()
            target.estimate_normals()
            
            # ICP registration
            trans_init = np.eye(4)
            
            reg_result = o3d.pipelines.registration.registration_icp(
                source, target, self.params.icp_distance_threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            if reg_result.fitness < self.params.icp_threshold:
                return None
            
            processing_time = time.time() - start_time
            
            # Estimate 2D projection for visualization
            h, w = template_data['depth'].shape
            corners_3d = np.array([
                [0, 0, np.mean(template_data['depth'][template_data['depth'] > 0])],
                [w, 0, np.mean(template_data['depth'][template_data['depth'] > 0])],
                [w, h, np.mean(template_data['depth'][template_data['depth'] > 0])],
                [0, h, np.mean(template_data['depth'][template_data['depth'] > 0])]
            ])
            
            corners_2d = self._project_3d_to_2d(corners_3d)
            
            return Detection(
                template_name=template_data['name'],
                confidence=reg_result.fitness,
                corners=corners_2d.reshape(-1, 1, 2),
                pose_6d={
                    'success': True,
                    'transformation': reg_result.transformation
                },
                modality=ModalityType.POINTCLOUD_ICP,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"ICP detection error: {e}")
            return None
    
    def _depth_to_pointcloud(self, depth: np.ndarray, offset_x: int = 0, 
                           offset_y: int = 0) -> np.ndarray:
        """Convert depth image to point cloud"""
        h, w = depth.shape
        
        # Create mesh grid
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Valid depth mask
        valid = depth > 0
        
        # Convert to 3D
        z = depth[valid]
        x = ((xx[valid] + offset_x) - self.depth_intrinsics.ppx) * z / self.depth_intrinsics.fx
        y = ((yy[valid] + offset_y) - self.depth_intrinsics.ppy) * z / self.depth_intrinsics.fy
        
        return np.column_stack((x, y, z))
    
    def _project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image plane"""
        if len(points_3d) == 0:
            return np.array([])
        
        # Project
        x_2d = (points_3d[:, 0] / points_3d[:, 2]) * self.depth_intrinsics.fx + self.depth_intrinsics.ppx
        y_2d = (points_3d[:, 1] / points_3d[:, 2]) * self.depth_intrinsics.fy + self.depth_intrinsics.ppy
        
        return np.column_stack((x_2d, y_2d))

class DetectionFusion:
    """Fuses multiple detection results"""
    
    @staticmethod
    def fuse_detections(detections: List[Detection], 
                       weights: Dict[ModalityType, float]) -> Optional[Detection]:
        """Fuse multiple detections using weighted voting"""
        if not detections:
            return None
        
        if len(detections) == 1:
            return detections[0]
        
        # Group detections by template
        template_detections = {}
        for det in detections:
            if det.template_name not in template_detections:
                template_detections[det.template_name] = []
            template_detections[det.template_name].append(det)
        
        # Fuse each template's detections
        fused_detections = []
        
        for template_name, dets in template_detections.items():
            # Calculate weighted confidence
            total_confidence = 0
            total_weight = 0
            
            for det in dets:
                weight = weights.get(det.modality, 0.25)
                total_confidence += det.confidence * weight
                total_weight += weight
            
            fused_confidence = total_confidence / total_weight if total_weight > 0 else np.mean([d.confidence for d in dets])
            
            # Average corners (simple fusion)
            all_corners = np.array([d.corners.reshape(-1, 2) for d in dets])
            fused_corners = np.mean(all_corners, axis=0)
            
            # Create fused detection
            fused_det = Detection(
                template_name=template_name,
                confidence=fused_confidence,
                corners=fused_corners.reshape(-1, 1, 2),
                pose_6d={'success': False},  # Will be calculated by pose estimator
                modality=ModalityType.MULTI_MODAL,
                processing_time=np.mean([d.processing_time for d in dets])
            )
            
            fused_detections.append(fused_det)
        
        # Return best fused detection
        if fused_detections:
            return max(fused_detections, key=lambda d: d.confidence)
        
        return None