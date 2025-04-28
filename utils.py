import cv2
import time
import numpy as np

class FPSCounter:
    def __init__(self):
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0
    
    def update(self):
        """Update FPS counter"""
        self.curr_frame_time = time.time()
        
        # Calculate FPS
        if self.prev_frame_time > 0:
            self.fps = 1 / (self.curr_frame_time - self.prev_frame_time)
        
        self.prev_frame_time = self.curr_frame_time
        
        return self.fps

def draw_face_detection(frame, face_box, prob, color, result_text):
    """Draw face detection and anti-spoofing results
    
    Args:
        frame: Original BGR frame
        face_box: Face bounding box [x_min, y_min, x_max, y_max]
        prob: Face detection confidence
        color: Rectangle color
        result_text: Text to display
    """
    x_min, y_min, x_max, y_max = face_box
    
    # Draw rectangle around face
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    
    # Display confidence score and anti-spoofing result
    cv2.putText(frame, f"Face: {prob:.2f}", (x_min, y_min - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.putText(frame, result_text, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_fps(frame, fps):
    """Draw FPS on frame
    
    Args:
        frame: Original BGR frame
        fps: Current FPS value
    """
    # Display FPS in top-left corner
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)