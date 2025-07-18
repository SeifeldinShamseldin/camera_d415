#!/usr/bin/env python3
"""
Featureless object detection engines for D415 Multi-Modal Detection System
Specialized detectors for objects with minimal texture features
"""

import cv2
import numpy as np
import time
from typing import Optional, List, Dict, Tuple
try:
    from scipy import ndimage
    from scipy.spatial.distance import correlation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..config import ModalityType, DetectionParams
from .engines import Detection

class ContourShapeDetector:
    """Contour-based detection using shape analysis"""
    
    def __init__(self, params: DetectionParams):
        self.params = params
    
    def detect(self, rgb: np.ndarray, template_data: Dict) -> Optional[Detection]:
        """Detect objects using contour shape matching"""
        try:
            start_time = time.time()
            
            # Extract template contour features
            template_contour_features = self._extract_contour_features(template_data['rgb'])
            if template_contour_features is None:
                return None
            
            # Extract scene contours
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            scene_contours = self._find_contours(gray)
            
            best_match = None
            best_score = float('inf')
            
            for contour in scene_contours:
                # Extract features for this contour
                contour_features = self._compute_contour_features(contour)
                if contour_features is None:
                    continue
                
                # Compare with template
                score = self._compare_contour_features(template_contour_features, contour_features)
                
                if score < self.params.contour_match_threshold and score < best_score:
                    best_score = score
                    
                    # Get bounding rectangle as detection corners
                    rect = cv2.boundingRect(contour)
                    x, y, w, h = rect
                    corners = np.array([
                        [x, y], [x+w, y], [x+w, y+h], [x, y+h]
                    ], dtype=np.float32).reshape(-1, 1, 2)
                    
                    best_match = {
                        'corners': corners,
                        'confidence': 1.0 - score,  # Convert distance to confidence
                        'contour': contour
                    }
            
            if best_match is None:
                return None
            
            processing_time = time.time() - start_time
            
            return Detection(
                template_name=template_data['name'],
                confidence=best_match['confidence'],
                corners=best_match['corners'],
                pose_6d={'success': False},
                modality=ModalityType.CONTOUR_SHAPE,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"Contour detection error: {e}")
            return None
    
    def _extract_contour_features(self, template_rgb: np.ndarray) -> Optional[Dict]:
        """Extract contour features from template"""
        gray = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)
        contours = self._find_contours(gray)
        
        if not contours:
            return None
        
        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        return self._compute_contour_features(largest_contour)
    
    def _find_contours(self, gray: np.ndarray) -> List:
        """Find contours in grayscale image"""
        try:
            if gray is None or gray.size == 0:
                return []
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive threshold for better contour detection
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by area
            filtered_contours = []
            for contour in contours:
                try:
                    area = cv2.contourArea(contour)
                    if area > self.params.contour_area_threshold:
                        filtered_contours.append(contour)
                except:
                    continue
            
            return filtered_contours
        except Exception:
            return []
    
    def _compute_contour_features(self, contour: np.ndarray) -> Optional[Dict]:
        """Compute feature descriptors for a contour"""
        try:
            # Hu moments (shape descriptors)
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                return None
            
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Normalize Hu moments (log scale)
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            
            # Additional shape features
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                return None
            
            # Compactness
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            
            # Aspect ratio
            rect = cv2.boundingRect(contour)
            aspect_ratio = float(rect[2]) / rect[3] if rect[3] != 0 else 0
            
            # Extent (object area / bounding box area)
            extent = area / (rect[2] * rect[3]) if (rect[2] * rect[3]) != 0 else 0
            
            return {
                'hu_moments': hu_moments,
                'compactness': compactness,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'area': area,
                'perimeter': perimeter
            }
            
        except Exception:
            return None
    
    def _compare_contour_features(self, template_features: Dict, scene_features: Dict) -> float:
        """Compare contour features and return distance"""
        # Hu moments comparison (primary descriptor)
        hu_distance = np.linalg.norm(
            template_features['hu_moments'] - scene_features['hu_moments']
        )
        
        # Geometric features comparison
        compactness_diff = abs(template_features['compactness'] - scene_features['compactness'])
        aspect_ratio_diff = abs(template_features['aspect_ratio'] - scene_features['aspect_ratio'])
        extent_diff = abs(template_features['extent'] - scene_features['extent'])
        
        # Weighted combination
        total_distance = (
            0.6 * hu_distance +
            0.15 * compactness_diff +
            0.15 * aspect_ratio_diff +
            0.1 * extent_diff
        )
        
        return total_distance

class DepthHistogramDetector:
    """Depth histogram matching for objects with characteristic depth profiles"""
    
    def __init__(self, params: DetectionParams):
        self.params = params
    
    def detect(self, depth: np.ndarray, template_data: Dict) -> Optional[Detection]:
        """Detect objects using depth histogram matching"""
        try:
            start_time = time.time()
            
            # Compute template depth histogram
            template_hist = self._compute_depth_histogram(template_data['depth'])
            if template_hist is None:
                return None
            
            # Sliding window approach for scene analysis
            template_h, template_w = template_data['depth'].shape
            best_match = None
            best_correlation = -1
            
            # Multi-scale search
            for scale in self.params.template_scales:
                scaled_h = int(template_h * scale)
                scaled_w = int(template_w * scale)
                
                if scaled_h >= depth.shape[0] or scaled_w >= depth.shape[1]:
                    continue
                
                # Slide window across the image
                for y in range(0, depth.shape[0] - scaled_h, 10):  # Step size 10 for efficiency
                    for x in range(0, depth.shape[1] - scaled_w, 10):
                        # Extract window
                        window = depth[y:y+scaled_h, x:x+scaled_w]
                        
                        # Compute histogram
                        window_hist = self._compute_depth_histogram(window)
                        if window_hist is None:
                            continue
                        
                        # Compare histograms
                        correlation = self._compare_histograms(template_hist, window_hist)
                        
                        if correlation > self.params.histogram_correlation_threshold and correlation > best_correlation:
                            best_correlation = correlation
                            
                            corners = np.array([
                                [x, y], [x+scaled_w, y], [x+scaled_w, y+scaled_h], [x, y+scaled_h]
                            ], dtype=np.float32).reshape(-1, 1, 2)
                            
                            best_match = {
                                'corners': corners,
                                'confidence': correlation,
                                'scale': scale
                            }
            
            if best_match is None:
                return None
            
            processing_time = time.time() - start_time
            
            return Detection(
                template_name=template_data['name'],
                confidence=best_match['confidence'],
                corners=best_match['corners'],
                pose_6d={'success': False},
                modality=ModalityType.DEPTH_HISTOGRAM,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"Depth histogram detection error: {e}")
            return None
    
    def _compute_depth_histogram(self, depth: np.ndarray) -> Optional[np.ndarray]:
        """Compute normalized depth histogram"""
        try:
            if depth is None or depth.size == 0:
                return None
                
            # Filter out zero depth values
            valid_depth = depth[depth > 0]
            
            if len(valid_depth) < 50:  # Need sufficient depth points
                return None
            
            # Check for valid range
            depth_min, depth_max = valid_depth.min(), valid_depth.max()
            if depth_min == depth_max:
                return None
            
            # Compute histogram
            hist, _ = np.histogram(valid_depth, bins=self.params.depth_bins, 
                                 range=(depth_min, depth_max))
            
            # Normalize
            hist = hist.astype(np.float32)
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                hist = hist / hist_sum
            else:
                return None
            
            return hist
        except Exception:
            return None
    
    def _compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compare two histograms using correlation"""
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

class GeometricPrimitivesDetector:
    """Detection using geometric primitives (circles, lines, rectangles)"""
    
    def __init__(self, params: DetectionParams):
        self.params = params
    
    def detect(self, rgb: np.ndarray, template_data: Dict) -> Optional[Detection]:
        """Detect objects using geometric primitive matching"""
        try:
            start_time = time.time()
            
            # Extract template primitives
            template_primitives = self._extract_primitives(template_data['rgb'])
            if not template_primitives:
                return None
            
            # Extract scene primitives
            scene_primitives = self._extract_primitives(rgb)
            if not scene_primitives:
                return None
            
            # Match primitives
            best_match = self._match_primitives(template_primitives, scene_primitives)
            
            if best_match is None:
                return None
            
            processing_time = time.time() - start_time
            
            return Detection(
                template_name=template_data['name'],
                confidence=best_match['confidence'],
                corners=best_match['corners'],
                pose_6d={'success': False},
                modality=ModalityType.GEOMETRIC_PRIMITIVES,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"Geometric primitives detection error: {e}")
            return None
    
    def _extract_primitives(self, rgb: np.ndarray) -> Dict:
        """Extract geometric primitives from image"""
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        primitives = {
            'circles': self._detect_circles(gray),
            'lines': self._detect_lines(edges),
            'rectangles': self._detect_rectangles(edges)
        }
        
        return primitives
    
    def _detect_circles(self, gray: np.ndarray) -> List:
        """Detect circles using Hough transform"""
        try:
            if gray is None or gray.size == 0:
                return []
                
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=self.params.hough_circle_param1,
                param2=self.params.hough_circle_param2,
                minRadius=self.params.min_circle_radius,
                maxRadius=self.params.max_circle_radius
            )
            
            if circles is not None and len(circles) > 0:
                circles = np.uint16(np.around(circles))
                return circles[0, :].tolist()
            return []
        except Exception:
            return []
    
    def _detect_lines(self, edges: np.ndarray) -> List:
        """Detect lines using Hough transform"""
        lines = cv2.HoughLines(edges, 1, np.pi/180, self.params.hough_line_threshold)
        
        if lines is not None:
            return lines.tolist()
        return []
    
    def _detect_rectangles(self, edges: np.ndarray) -> List:
        """Detect rectangles using contour analysis"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle (4 corners)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > self.params.contour_area_threshold:
                    rectangles.append(approx.reshape(-1, 2).tolist())
        
        return rectangles
    
    def _match_primitives(self, template_primitives: Dict, scene_primitives: Dict) -> Optional[Dict]:
        """Match geometric primitives between template and scene"""
        # Simple matching based on primitive counts and properties
        template_circles = len(template_primitives['circles'])
        template_lines = len(template_primitives['lines'])
        template_rectangles = len(template_primitives['rectangles'])
        
        scene_circles = len(scene_primitives['circles'])
        scene_lines = len(scene_primitives['lines'])
        scene_rectangles = len(scene_primitives['rectangles'])
        
        # Calculate similarity score
        circle_score = 1.0 - abs(template_circles - scene_circles) / max(template_circles + scene_circles, 1)
        line_score = 1.0 - abs(template_lines - scene_lines) / max(template_lines + scene_lines, 1)
        rect_score = 1.0 - abs(template_rectangles - scene_rectangles) / max(template_rectangles + scene_rectangles, 1)
        
        overall_score = (circle_score + line_score + rect_score) / 3.0
        
        if overall_score < 0.6:  # Threshold for geometric similarity
            return None
        
        # For simplicity, return first found rectangle as detection
        if scene_primitives['rectangles']:
            rect = scene_primitives['rectangles'][0]
            corners = np.array(rect, dtype=np.float32).reshape(-1, 1, 2)
            
            return {
                'corners': corners,
                'confidence': overall_score
            }
        
        # If no rectangles, use bounding box of all primitives
        if scene_primitives['circles']:
            circle = scene_primitives['circles'][0]
            x, y, r = circle
            corners = np.array([
                [x-r, y-r], [x+r, y-r], [x+r, y+r], [x-r, y+r]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            return {
                'corners': corners,
                'confidence': overall_score
            }
        
        return None

class SurfaceNormalsDetector:
    """Surface normal analysis for curved featureless objects"""
    
    def __init__(self, params: DetectionParams, depth_intrinsics):
        self.params = params
        self.depth_intrinsics = depth_intrinsics
    
    def detect(self, depth: np.ndarray, template_data: Dict) -> Optional[Detection]:
        """Detect objects using surface normal patterns"""
        try:
            start_time = time.time()
            
            # Compute template surface normals
            template_normals = self._compute_surface_normals(template_data['depth'])
            if template_normals is None:
                return None
            
            # Compute scene surface normals
            scene_normals = self._compute_surface_normals(depth)
            if scene_normals is None:
                return None
            
            # Template matching on normal maps
            result = cv2.matchTemplate(scene_normals, template_normals, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val < self.params.normal_correlation_threshold:
                return None
            
            # Extract detection region
            h, w = template_normals.shape[:2]
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
                modality=ModalityType.SURFACE_NORMALS,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"Surface normals detection error: {e}")
            return None
    
    def _compute_surface_normals(self, depth: np.ndarray) -> Optional[np.ndarray]:
        """Compute surface normals from depth image"""
        try:
            # Convert depth to point cloud
            h, w = depth.shape
            
            # Create coordinate grids
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            
            # Valid depth mask
            valid = depth > 0
            
            if np.sum(valid) < 100:
                return None
            
            # Convert to 3D points
            z = depth.astype(np.float32)
            x = (xx - self.depth_intrinsics.ppx) * z / self.depth_intrinsics.fx
            y = (yy - self.depth_intrinsics.ppy) * z / self.depth_intrinsics.fy
            
            # Compute gradients
            grad_x = cv2.Sobel(z, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(z, cv2.CV_64F, 0, 1, ksize=3)
            
            # Compute normal vectors
            # Normal = (-grad_x, -grad_y, 1)
            normal_x = -grad_x
            normal_y = -grad_y
            normal_z = np.ones_like(grad_x)
            
            # Normalize
            magnitude = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
            magnitude[magnitude == 0] = 1e-7
            
            normal_x /= magnitude
            normal_y /= magnitude
            normal_z /= magnitude
            
            # Create normal map image (encode normals as RGB)
            normal_map = np.zeros((h, w, 3), dtype=np.uint8)
            normal_map[:,:,0] = ((normal_x + 1) * 127.5).astype(np.uint8)  # Red channel
            normal_map[:,:,1] = ((normal_y + 1) * 127.5).astype(np.uint8)  # Green channel
            normal_map[:,:,2] = ((normal_z + 1) * 127.5).astype(np.uint8)  # Blue channel
            
            # Convert to grayscale for template matching
            normal_gray = cv2.cvtColor(normal_map, cv2.COLOR_BGR2GRAY)
            
            return normal_gray
            
        except Exception:
            return None