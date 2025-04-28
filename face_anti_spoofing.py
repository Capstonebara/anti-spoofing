import cv2
import numpy as np
import os
import sys

# Import modules
from face_detector import FaceDetector
from anti_spoofing import AntiSpoofingPredictor
from utils import FPSCounter, draw_face_detection, draw_fps

def main():
    # Initialize components
    face_detector = FaceDetector()
    anti_spoofing = AntiSpoofingPredictor()
    fps_counter = FPSCounter()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        exit()
    
    print("Starting real-time face anti-spoofing detection. Press 'q' to quit.")
    
    # Real-time detection loop
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Update FPS
        fps = fps_counter.update()
        
        # Detect faces
        boxes, probs = face_detector.detect_faces(frame)
        
        # Process each detected face
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < 0.8:  # Confidence threshold
                    continue
                    
                # Convert box coordinates to integers
                box = box.astype(int)
                x_min, y_min, x_max, y_max = box
                
                # Create bbox for anti-spoofing
                image_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                
                # Preprocess face for anti-spoofing
                face_img = anti_spoofing.preprocess_face(frame, image_bbox)
                
                # Run anti-spoofing prediction
                prediction = anti_spoofing.predict(face_img)
                
                # Get result
                label = np.argmax(prediction)
                value = prediction[0][label]
                
                # Set color and result text based on prediction
                if label == 1:
                    # Real face - green
                    color = (0, 255, 0)
                    result_text = f"Real Face: {value:.2f}"
                else:
                    # Fake face - red
                    color = (0, 0, 255)
                    result_text = f"Fake Face: {value:.2f}"
                    
                # Draw detection results
                draw_face_detection(frame, box, prob, color, result_text)
        
        # Draw FPS counter
        draw_fps(frame, fps)
        
        # Display the output frame
        cv2.imshow("Face Anti-Spoofing", frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()