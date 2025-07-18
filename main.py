#!/usr/bin/env python3
"""
Main application for D415 Multi-Modal Adaptive Detection System
Handles UI, user interaction, and main processing loop
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple

from .core.detector import AdaptiveMultiModalDetector
from .config import load_config, SystemConfig

class DetectionApp:
    """Main application class for the detection system"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize detector
        self.detector = AdaptiveMultiModalDetector(self.config)
        
        # UI state
        self.creating_template = False
        self.roi_start = None
        self.roi_end = None
        self.current_frame_rgb = None
        self.current_frame_depth = None
        self.current_frame_ir = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
    
    def initialize(self) -> bool:
        """Initialize the application"""
        print("\n" + "="*70)
        print("🚀 D415 Multi-Modal Adaptive Detection System")
        print("   Fraunhofer Refactored Version")
        print("="*70)
        
        # Initialize camera
        if not self.detector.initialize_camera():
            return False
        
        # Load existing templates
        self.detector.load_templates()
        
        return True
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for template creation"""
        if not self.creating_template:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_start = (x, y)
            self.roi_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.roi_start:
            self.roi_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_end = (x, y)
    
    def _handle_keyboard(self, key: int) -> bool:
        """Handle keyboard input"""
        if key == ord('q'):
            return False
        elif key == ord('t'):
            self._start_template_creation()
        elif key == ord('c') and self.creating_template:
            self._cancel_template_creation()
        elif key == ord('s') and self.creating_template and self.roi_start and self.roi_end:
            self._save_template()
        elif key == ord('r'):
            self.detector.load_templates()
            print("✅ Templates reloaded")
        
        return True
    
    def _start_template_creation(self):
        """Start template creation mode"""
        self.creating_template = True
        self.roi_start = None
        self.roi_end = None
        print("\n" + "="*60)
        print("📋 TEMPLATE CREATION MODE ACTIVATED")
        print("="*60)
        print("Instructions:")
        print("1. Draw ROI around object with mouse")
        print("2. Press 's' to save template")
        print("3. Press 'c' to cancel")
        print("="*60)
    
    def _cancel_template_creation(self):
        """Cancel template creation"""
        self.creating_template = False
        self.roi_start = None
        self.roi_end = None
        print("❌ Template creation cancelled")
    
    def _save_template(self):
        """Save the created template"""
        if not (self.roi_start and self.roi_end and 
                self.current_frame_rgb is not None and 
                self.current_frame_depth is not None):
            print("⚠️  No valid frame or ROI data")
            return
        
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        if x2 - x1 > 20 and y2 - y1 > 20:
            templates = self.detector.template_manager.get_all_templates()
            template_name = f"template_{len(templates) + 1}"
            roi = (x1, y1, x2 - x1, y2 - y1)
            
            if self.detector.create_template(
                template_name, 
                self.current_frame_rgb,
                self.current_frame_depth, 
                self.current_frame_ir, 
                roi
            ):
                print(f"\n✅ Template '{template_name}' created successfully!")
                print(f"📁 Saved to: {self.config.template_dir}/{template_name}/")
                templates = self.detector.template_manager.get_all_templates()
                print(f"📊 Total templates: {len(templates)}")
                print("🔄 Template will be used for detection\n")
            
            self.creating_template = False
            self.roi_start = None
            self.roi_end = None
        else:
            print("⚠️  ROI too small")
    
    def _draw_template_creation_ui(self, frame: np.ndarray):
        """Draw template creation UI"""
        cv2.putText(frame, "Creating Template - Draw ROI",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        if self.roi_start and self.roi_end:
            cv2.rectangle(frame, self.roi_start, self.roi_end, (0, 255, 0), 2)
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed
    
    def run(self):
        """Main processing loop"""
        # Setup window and mouse callback
        window_name = 'D415 Multi-Modal Detection - Fraunhofer'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        print("\n📹 Streaming started!")
        print("Commands:")
        print("  t - Create new template")
        print("  s - Save template selection")
        print("  r - Reload templates")
        print("  c - Cancel template creation")
        print("  q - Quit")
        print()
        
        try:
            while True:
                # Get frames
                frames = self.detector.pipeline.wait_for_frames()
                aligned_frames = self.detector.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                # Get IR frame if enabled
                ir_frame = None
                if self.config.enable_ir:
                    try:
                        ir_frame = aligned_frames.get_infrared_frame(1)
                    except:
                        pass
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                self.current_frame_rgb = np.asanyarray(color_frame.get_data())
                self.current_frame_depth = np.asanyarray(depth_frame.get_data())
                self.current_frame_ir = (
                    np.asanyarray(ir_frame.get_data()) if ir_frame else None
                )
                
                display_frame = self.current_frame_rgb.copy()
                
                if not self.creating_template:
                    # Detection mode
                    detections = self.detector.detect_adaptive(
                        self.current_frame_rgb,
                        self.current_frame_depth,
                        self.current_frame_ir
                    )
                    
                    # Draw detections
                    for detection in detections:
                        self.detector.draw_detection(display_frame, detection)
                    
                    # Draw status overlay
                    self.detector.draw_status_overlay(display_frame, self.fps, detections)
                else:
                    # Template creation mode
                    self._draw_template_creation_ui(display_frame)
                
                # Update performance metrics
                self._update_fps()
                
                # Display frame
                cv2.imshow(window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard(key):
                    break
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources"""
        self.detector.cleanup()
        cv2.destroyAllWindows()
        
        # Print final statistics
        stats = self.detector.get_statistics()
        print("\n" + "="*70)
        print("📊 Session Statistics:")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Average FPS: {self.fps:.1f}")
        print(f"  Total detections: {stats['total_detections']}")
        print(f"  Modality usage:")
        for modality, count in stats['modality_usage'].items():
            if count > 0:
                print(f"    {modality.value}: {count}")
        print("="*70)
        print("✅ Session complete!")

def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="D415 Multi-Modal Adaptive Detection System"
    )
    parser.add_argument(
        '--config', '-c', 
        help="Path to configuration file",
        default=None
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Override config if needed
    config = load_config(args.config)
    if args.verbose:
        config.verbose = True
    
    # Create and run application
    app = DetectionApp()
    app.config = config
    
    if app.initialize():
        app.run()
    else:
        print("❌ Failed to initialize application")

if __name__ == "__main__":
    main()