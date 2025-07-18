#!/usr/bin/env python3
"""
6D pose estimation utilities for D415 Multi-Modal Detection System
Handles pose estimation from 2D detections and 3D transformations
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple

class PoseEstimator:
    """6D pose estimation from 2D corners and depth information"""
    
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def estimate_pose_from_corners(self, corners_2d: np.ndarray, 
                                  object_size: float = 100.0) -> Dict:
        """Estimate 6D pose from 2D corners using PnP"""
        try:
            # Define 3D object points (square object in mm)
            half_size = object_size / 2
            object_3d = np.array([
                [-half_size, -half_size, 0],
                [half_size, -half_size, 0],
                [half_size, half_size, 0],
                [-half_size, half_size, 0]
            ], dtype=np.float32)
            
            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                object_3d, corners_2d,
                self.camera_matrix, self.dist_coeffs
            )
            
            if not success:
                return {'success': False}
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Extract Euler angles
            euler_angles = self._rotation_matrix_to_euler(rotation_matrix)
            
            return {
                'success': True,
                'position': tvec.flatten().tolist(),
                'rotation': np.degrees(euler_angles).tolist(),
                'rvec': rvec,
                'tvec': tvec,
                'rotation_matrix': rotation_matrix
            }
            
        except Exception as e:
            print(f"Pose estimation error: {e}")
            return {'success': False}
    
    def estimate_pose_from_transformation(self, transformation_matrix: np.ndarray) -> Dict:
        """Extract 6D pose from 4x4 transformation matrix (from ICP)"""
        try:
            rotation_matrix = transformation_matrix[:3, :3]
            translation = transformation_matrix[:3, 3]
            
            # Extract Euler angles
            euler_angles = self._rotation_matrix_to_euler(rotation_matrix)
            
            return {
                'success': True,
                'position': translation.tolist(),
                'rotation': np.degrees(euler_angles).tolist(),
                'transformation': transformation_matrix
            }
            
        except Exception as e:
            print(f"Transformation pose estimation error: {e}")
            return {'success': False}
    
    def _rotation_matrix_to_euler(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (XYZ order)"""
        # Extract Euler angles using ZYX convention (roll, pitch, yaw)
        sy = np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])  # Roll
            y = np.arctan2(-rotation_matrix[2,0], sy)                   # Pitch
            z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])  # Yaw
        else:
            x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def project_points_to_image(self, points_3d: np.ndarray, 
                               rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image plane"""
        try:
            points_2d, _ = cv2.projectPoints(
                points_3d, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
            return points_2d.reshape(-1, 2)
        except:
            return np.array([])
    
    def get_coordinate_axes(self, scale: float = 50.0) -> np.ndarray:
        """Get 3D coordinate axes for visualization"""
        return np.array([
            [0, 0, 0],      # Origin
            [scale, 0, 0],  # X-axis (Red)
            [0, scale, 0],  # Y-axis (Green)
            [0, 0, -scale]  # Z-axis (Blue)
        ], dtype=np.float32)

class PoseVisualizer:
    """Visualization utilities for 6D pose"""
    
    def __init__(self, pose_estimator: PoseEstimator):
        self.pose_estimator = pose_estimator
    
    def draw_pose_axes(self, frame: np.ndarray, pose_6d: Dict, 
                      scale: float = 50.0, thickness: int = 3):
        """Draw coordinate axes for pose visualization"""
        if not pose_6d.get('success', False):
            return
        
        try:
            # Get 3D axes points
            axes_3d = self.pose_estimator.get_coordinate_axes(scale)
            
            # Project to 2D
            axes_2d = self.pose_estimator.project_points_to_image(
                axes_3d,
                pose_6d.get('rvec', np.zeros((3, 1))),
                pose_6d.get('tvec', np.zeros((3, 1)))
            )
            
            if len(axes_2d) < 4:
                return
            
            # Check for valid coordinates and convert to int
            if not np.all(np.isfinite(axes_2d)):
                return
                
            axes_2d = np.round(axes_2d).astype(int)
            
            # Ensure all points are within image bounds
            h, w = frame.shape[:2]
            valid_points = []
            for point in axes_2d:
                if 0 <= point[0] < w and 0 <= point[1] < h:
                    valid_points.append(tuple(point))
                else:
                    return  # Skip drawing if any point is out of bounds
            
            if len(valid_points) < 4:
                return
            
            origin = valid_points[0]
            
            # Draw axes
            cv2.arrowedLine(frame, origin, valid_points[1], (0, 0, 255), thickness)  # X - Red
            cv2.arrowedLine(frame, origin, valid_points[2], (0, 255, 0), thickness)  # Y - Green
            cv2.arrowedLine(frame, origin, valid_points[3], (255, 0, 0), thickness)  # Z - Blue
            
        except Exception as e:
            print(f"Pose axes drawing error: {e}")
    
    def draw_pose_info(self, frame: np.ndarray, pose_6d: Dict, 
                      position: Tuple[int, int], color: Tuple[int, int, int] = (255, 255, 255)):
        """Draw pose information text"""
        if not pose_6d.get('success', False):
            return
        
        try:
            x, y = position
            pos = pose_6d['position']
            rot = pose_6d['rotation']
            
            # Position text
            pos_text = f"XYZ: {pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f}mm"
            cv2.putText(frame, pos_text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Rotation text
            rot_text = f"RPY: {rot[0]:.0f},{rot[1]:.0f},{rot[2]:.0f}°"
            cv2.putText(frame, rot_text, (x, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        except Exception as e:
            print(f"Pose info drawing error: {e}")
    
    def draw_object_wireframe(self, frame: np.ndarray, pose_6d: Dict, 
                             object_size: float = 100.0, color: Tuple[int, int, int] = (0, 255, 255)):
        """Draw 3D wireframe of detected object"""
        if not pose_6d.get('success', False):
            return
        
        try:
            # Define 3D wireframe points (cube)
            half_size = object_size / 2
            wireframe_3d = np.array([
                # Bottom face
                [-half_size, -half_size, 0],
                [half_size, -half_size, 0],
                [half_size, half_size, 0],
                [-half_size, half_size, 0],
                # Top face
                [-half_size, -half_size, -object_size],
                [half_size, -half_size, -object_size],
                [half_size, half_size, -object_size],
                [-half_size, half_size, -object_size]
            ], dtype=np.float32)
            
            # Project to 2D
            wireframe_2d = self.pose_estimator.project_points_to_image(
                wireframe_3d,
                pose_6d.get('rvec', np.zeros((3, 1))),
                pose_6d.get('tvec', np.zeros((3, 1)))
            )
            
            if len(wireframe_2d) < 8:
                return
            
            # Check for valid coordinates and convert to int
            if not np.all(np.isfinite(wireframe_2d)):
                return
                
            wireframe_2d = np.round(wireframe_2d).astype(int)
            
            # Draw bottom face
            for i in range(4):
                pt1 = tuple(wireframe_2d[i])
                pt2 = tuple(wireframe_2d[(i + 1) % 4])
                cv2.line(frame, pt1, pt2, color, 2)
            
            # Draw top face
            for i in range(4, 8):
                pt1 = tuple(wireframe_2d[i])
                pt2 = tuple(wireframe_2d[4 + ((i - 4 + 1) % 4)])
                cv2.line(frame, pt1, pt2, color, 2)
            
            # Draw vertical edges
            for i in range(4):
                pt1 = tuple(wireframe_2d[i])
                pt2 = tuple(wireframe_2d[i + 4])
                cv2.line(frame, pt1, pt2, color, 2)
            
        except Exception as e:
            print(f"Wireframe drawing error: {e}")

class PoseFilter:
    """Temporal filtering for pose stability"""
    
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha  # Filter coefficient (0 = no filtering, 1 = no update)
        self.previous_pose = None
        self.pose_history = []
        self.max_history = 10
    
    def filter_pose(self, pose_6d: Dict) -> Dict:
        """Apply temporal filtering to pose"""
        if not pose_6d.get('success', False):
            return pose_6d
        
        current_position = np.array(pose_6d['position'])
        current_rotation = np.array(pose_6d['rotation'])
        
        if self.previous_pose is None:
            # First pose, no filtering
            filtered_pose = pose_6d.copy()
        else:
            # Apply exponential moving average
            prev_position = np.array(self.previous_pose['position'])
            prev_rotation = np.array(self.previous_pose['rotation'])
            
            # Filter position
            filtered_position = self.alpha * prev_position + (1 - self.alpha) * current_position
            
            # Filter rotation (handle angle wrapping)
            rotation_diff = current_rotation - prev_rotation
            rotation_diff = np.where(rotation_diff > 180, rotation_diff - 360, rotation_diff)
            rotation_diff = np.where(rotation_diff < -180, rotation_diff + 360, rotation_diff)
            filtered_rotation = prev_rotation + (1 - self.alpha) * rotation_diff
            
            # Normalize angles
            filtered_rotation = np.where(filtered_rotation > 180, filtered_rotation - 360, filtered_rotation)
            filtered_rotation = np.where(filtered_rotation < -180, filtered_rotation + 360, filtered_rotation)
            
            filtered_pose = pose_6d.copy()
            filtered_pose['position'] = filtered_position.tolist()
            filtered_pose['rotation'] = filtered_rotation.tolist()
        
        self.previous_pose = filtered_pose
        
        # Add to history
        self.pose_history.append(filtered_pose)
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
        
        return filtered_pose
    
    def get_pose_stability(self) -> float:
        """Calculate pose stability metric (0-1, higher is more stable)"""
        if len(self.pose_history) < 3:
            return 0.0
        
        try:
            # Calculate variance in recent poses
            recent_positions = np.array([p['position'] for p in self.pose_history[-5:]])
            recent_rotations = np.array([p['rotation'] for p in self.pose_history[-5:]])
            
            position_variance = np.mean(np.var(recent_positions, axis=0))
            rotation_variance = np.mean(np.var(recent_rotations, axis=0))
            
            # Convert to stability metric (inverse of variance, clamped to [0,1])
            position_stability = 1.0 / (1.0 + position_variance / 100.0)
            rotation_stability = 1.0 / (1.0 + rotation_variance / 100.0)
            
            return (position_stability + rotation_stability) / 2.0
            
        except:
            return 0.0
    
    def reset(self):
        """Reset filter state"""
        self.previous_pose = None
        self.pose_history = []