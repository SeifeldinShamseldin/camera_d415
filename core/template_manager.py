#!/usr/bin/env python3
"""
Template management module for D415 Multi-Modal Detection System
Handles template creation, storage, loading, and analysis
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..config import ModalityType, DetectionParams

@dataclass
class TemplateAnalysis:
    """Template characteristics analysis"""
    has_texture: bool
    sift_features: int
    edge_density: float
    depth_variance: float
    ir_contrast: float
    optimal_modalities: List[ModalityType]
    weights: Dict[ModalityType, float]

class MultiModalTemplate:
    """Stores multi-modal template data"""
    def __init__(self, name: str):
        self.name = name
        self.rgb = None
        self.depth = None
        self.ir = None
        self.pointcloud = None
        self.analysis = None
        self.sift_features = None
        self.metadata = {}

class TemplateAnalyzer:
    """Analyzes template characteristics to determine optimal detection strategy"""
    
    def __init__(self, params: DetectionParams):
        self.params = params
        self.sift_detector = cv2.SIFT_create(nfeatures=params.sift_features)
    
    def analyze_template(self, rgb: np.ndarray, depth: np.ndarray, 
                        ir: Optional[np.ndarray] = None) -> TemplateAnalysis:
        """Analyze template characteristics to determine optimal detection strategy"""
        
        # RGB texture analysis
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        kp_sift, _ = self.sift_detector.detectAndCompute(gray, None)
        sift_count = len(kp_sift) if kp_sift else 0
        
        # Edge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Depth variance (indicates 3D structure)
        depth_valid = depth[depth > 0]
        depth_variance = np.var(depth_valid) if len(depth_valid) > 0 else 0
        
        # IR contrast
        ir_contrast = np.std(ir) / (np.mean(ir) + 1e-6) if ir is not None else 0
        
        # Determine characteristics
        has_texture = sift_count > self.params.texture_sift_threshold
        has_edges = edge_density > self.params.edge_density_threshold
        has_3d_structure = depth_variance > self.params.depth_variance_threshold
        has_ir_features = ir_contrast > self.params.ir_contrast_threshold
        
        # Additional characteristics for featureless objects
        is_featureless = sift_count < 15  # More generous threshold
        has_geometric_shape = edge_density > 0.01  # Lower threshold for edges
        has_depth_profile = depth_variance > 50   # Lower threshold for depth structure
        
        # Select optimal modalities and weights
        modalities = []
        weights = {}
        
        if has_texture:
            modalities.append(ModalityType.RGB_SIFT)
            weights[ModalityType.RGB_SIFT] = 0.4
        else:
            modalities.append(ModalityType.RGB_TEMPLATE)
            weights[ModalityType.RGB_TEMPLATE] = 0.2
            
        if has_ir_features or not has_texture:
            modalities.append(ModalityType.IR_TEMPLATE)
            weights[ModalityType.IR_TEMPLATE] = 0.25 if not has_texture else 0.15
            
        if has_edges and has_3d_structure:
            modalities.append(ModalityType.DEPTH_EDGE)
            weights[ModalityType.DEPTH_EDGE] = 0.15
            
        if has_3d_structure:
            modalities.append(ModalityType.POINTCLOUD_ICP)
            weights[ModalityType.POINTCLOUD_ICP] = 0.2
        
        # Add featureless object detection modalities
        if is_featureless:
            # Contour-based detection for shape analysis
            if has_geometric_shape:
                modalities.append(ModalityType.CONTOUR_SHAPE)
                weights[ModalityType.CONTOUR_SHAPE] = 0.25
            
            # Depth histogram for objects with characteristic depth profiles
            if has_depth_profile:
                modalities.append(ModalityType.DEPTH_HISTOGRAM)
                weights[ModalityType.DEPTH_HISTOGRAM] = 0.2
            
            # Geometric primitives for simple shapes
            modalities.append(ModalityType.GEOMETRIC_PRIMITIVES)
            weights[ModalityType.GEOMETRIC_PRIMITIVES] = 0.15
            
            # Surface normals for curved objects
            if has_3d_structure:
                modalities.append(ModalityType.SURFACE_NORMALS)
                weights[ModalityType.SURFACE_NORMALS] = 0.2
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return TemplateAnalysis(
            has_texture=has_texture,
            sift_features=sift_count,
            edge_density=edge_density,
            depth_variance=depth_variance,
            ir_contrast=ir_contrast,
            optimal_modalities=modalities,
            weights=weights
        )

class TemplateManager:
    """Manages template storage, loading, and creation"""
    
    def __init__(self, template_dir: str, params: DetectionParams):
        self.template_dir = template_dir
        self.params = params
        self.templates: Dict[str, MultiModalTemplate] = {}
        self.analyzer = TemplateAnalyzer(params)
        
        # Create template directory
        os.makedirs(template_dir, exist_ok=True)
    
    def create_template(self, name: str, rgb: np.ndarray, depth: np.ndarray, 
                       ir: Optional[np.ndarray], roi: Tuple[int, int, int, int],
                       depth_intrinsics=None) -> bool:
        """Create multi-modal template from ROI"""
        try:
            x, y, w, h = roi
            
            # Extract ROIs
            rgb_roi = rgb[y:y+h, x:x+w]
            depth_roi = depth[y:y+h, x:x+w]
            ir_roi = ir[y:y+h, x:x+w] if ir is not None else None
            
            # Create template
            template = MultiModalTemplate(name)
            template.rgb = rgb_roi
            template.depth = depth_roi
            template.ir = ir_roi
            
            # Analyze template
            template.analysis = self.analyzer.analyze_template(rgb_roi, depth_roi, ir_roi)
            
            # Extract SIFT features if textured
            if template.analysis.has_texture:
                gray = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2GRAY)
                kp, desc = self.analyzer.sift_detector.detectAndCompute(gray, None)
                template.sift_features = {'keypoints': kp, 'descriptors': desc}
            
            # Generate point cloud template if depth available
            if depth_intrinsics and np.sum(depth_roi > 0) > 100:
                pc_points = self._depth_to_pointcloud(depth_roi, x, y, depth_intrinsics)
                if len(pc_points) > 0:
                    template.pointcloud = pc_points
            
            # Store metadata
            template.metadata = {
                'roi': roi,
                'timestamp': datetime.now().isoformat(),
                'analysis': {
                    'sift_features': template.analysis.sift_features,
                    'edge_density': template.analysis.edge_density,
                    'depth_variance': template.analysis.depth_variance,
                    'optimal_modalities': [m.value for m in template.analysis.optimal_modalities]
                }
            }
            
            # Save template
            self.templates[name] = template
            self._save_template(template)
            
            print(f"✅ Template '{name}' created:")
            print(f"   Texture: {'Yes' if template.analysis.has_texture else 'No'}")
            print(f"   SIFT features: {template.analysis.sift_features}")
            print(f"   Optimal modalities: {[m.value for m in template.analysis.optimal_modalities]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Template creation failed: {e}")
            return False
    
    def _save_template(self, template: MultiModalTemplate):
        """Save template to disk"""
        template_path = os.path.join(self.template_dir, template.name)
        os.makedirs(template_path, exist_ok=True)
        
        # Save images
        cv2.imwrite(os.path.join(template_path, "rgb.png"), template.rgb)
        if template.ir is not None:
            cv2.imwrite(os.path.join(template_path, "ir.png"), template.ir)
        np.save(os.path.join(template_path, "depth.npy"), template.depth)
        
        # Save metadata
        with open(os.path.join(template_path, "metadata.json"), 'w') as f:
            json.dump(template.metadata, f, indent=2)
        
        # Save point cloud if available
        if template.pointcloud is not None:
            np.save(os.path.join(template_path, "pointcloud.npy"), template.pointcloud)
    
    def load_templates(self):
        """Load all templates from disk"""
        if not os.path.exists(self.template_dir):
            print(f"Template directory {self.template_dir} does not exist")
            return
            
        loaded_count = 0
        for template_name in os.listdir(self.template_dir):
            template_path = os.path.join(self.template_dir, template_name)
            if os.path.isdir(template_path):
                try:
                    template = self._load_single_template(template_name, template_path)
                    if template:
                        self.templates[template_name] = template
                        loaded_count += 1
                        print(f"✅ Loaded template: {template_name}")
                except Exception as e:
                    print(f"⚠️  Failed to load template {template_name}: {e}")
        
        print(f"📚 Total templates loaded: {loaded_count}")
    
    def _load_single_template(self, template_name: str, template_path: str) -> Optional[MultiModalTemplate]:
        """Load a single template from disk"""
        template = MultiModalTemplate(template_name)
        
        # Load images
        rgb_path = os.path.join(template_path, "rgb.png")
        if not os.path.exists(rgb_path):
            return None
            
        template.rgb = cv2.imread(rgb_path)
        
        ir_path = os.path.join(template_path, "ir.png")
        template.ir = cv2.imread(ir_path, 0) if os.path.exists(ir_path) else None
        
        depth_path = os.path.join(template_path, "depth.npy")
        if os.path.exists(depth_path):
            template.depth = np.load(depth_path)
        else:
            return None
        
        # Load metadata
        metadata_path = os.path.join(template_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                template.metadata = json.load(f)
        
        # Re-analyze template
        template.analysis = self.analyzer.analyze_template(
            template.rgb, template.depth, template.ir
        )
        
        # Re-extract SIFT features
        if template.analysis.has_texture:
            gray = cv2.cvtColor(template.rgb, cv2.COLOR_BGR2GRAY)
            kp, desc = self.analyzer.sift_detector.detectAndCompute(gray, None)
            template.sift_features = {'keypoints': kp, 'descriptors': desc}
        
        # Load point cloud if available
        pc_path = os.path.join(template_path, "pointcloud.npy")
        if os.path.exists(pc_path):
            template.pointcloud = np.load(pc_path)
        
        return template
    
    def get_template_data(self, template_name: str) -> Optional[Dict]:
        """Get template data for detection engines"""
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        return {
            'name': template.name,
            'rgb': template.rgb,
            'depth': template.depth,
            'ir': template.ir,
            'pointcloud': template.pointcloud,
            'sift_features': template.sift_features,
            'analysis': template.analysis
        }
    
    def get_all_templates(self) -> Dict[str, MultiModalTemplate]:
        """Get all loaded templates"""
        return self.templates
    
    def delete_template(self, name: str) -> bool:
        """Delete a template"""
        try:
            if name in self.templates:
                del self.templates[name]
            
            template_path = os.path.join(self.template_dir, name)
            if os.path.exists(template_path):
                import shutil
                shutil.rmtree(template_path)
            
            print(f"✅ Template '{name}' deleted")
            return True
        except Exception as e:
            print(f"❌ Failed to delete template '{name}': {e}")
            return False
    
    def _depth_to_pointcloud(self, depth: np.ndarray, offset_x: int, offset_y: int,
                           depth_intrinsics) -> np.ndarray:
        """Convert depth image to point cloud"""
        h, w = depth.shape
        
        # Create mesh grid
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Valid depth mask
        valid = depth > 0
        
        # Convert to 3D
        z = depth[valid]
        x = ((xx[valid] + offset_x) - depth_intrinsics.ppx) * z / depth_intrinsics.fx
        y = ((yy[valid] + offset_y) - depth_intrinsics.ppy) * z / depth_intrinsics.fy
        
        return np.column_stack((x, y, z))